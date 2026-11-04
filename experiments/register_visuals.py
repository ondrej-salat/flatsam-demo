"""Render the mask->H registration internals from real runs, for documentation.

Every panel is produced by calling the deployed code on real data — no mock
inputs, no hand-drawn diagrams:

* the SAM3 masks come from an existing ``hough_symmetry_only`` run (the same
  mask stream the tracker consumes),
* the registration panels call :func:`flatsam2.mask_register.register_mask` and
  :func:`flatsam2.mask_align._gn_align` themselves,
* the fusion panel reads the per-frame covariances a
  ``corner_fusion_register_final`` run already wrote to ``<seq>_debug.npz``.

Panels written (PNG, one file each):

``inputs``    template silhouette + a later frame with its SAM3 mask
``sdf``       the signed-distance field the GN objective samples, with the
              warped template boundary on it and the residual histogram
``gn``        control-point iterates of one damped Gauss-Newton solve from a
              perturbed start, plus its cost curve
``propose``   every second-moment proposal of the angle sweep, scored
``jacobian``  the boundary samples coloured by how strongly each constrains
              each corner — the matrix behind the covariance, drawn
``verify``    the winning pose against its three cyclic relabelings
``fusion``    both branches' corner estimates with covariance ellipses and the
              per-corner squared Mahalanobis disagreement

CPU-only; needs cv2, so submit it rather than running on a login node::

    python slurm/submit_cmd.py --gpus 0 --no-mail -- \\
        python experiments/register_visuals.py --out-dir <dir>
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from flatsam2.corner_covariance import corners_to_theta, theta_to_corners
from flatsam2.datasets import corners_to_mask, make_gt_loader, make_image_paths, warp_corners
from flatsam2.mask_align import _corner_cov_from_boundary_fit, _gn_align, _residuals
from flatsam2.mask_register import (
    RegisterConfig,
    RegistrationTemplate,
    _overlap,
    _sdf_full,
    moment_affine_inits,
    register_mask,
    second_moments,
    verification_score,
)
from flatsam2.path_config import paths

# Colours shared with the write-up: keypoint branch, mask branch, fusion.
C_KP = "#C08A2E"
C_MASK = "#1188A0"
C_FUSED = "#7A68C0"
C_GT = "#3FBF6F"
C_BAD = "#C4553F"

_QUAD_ORDER = [0, 1, 2, 3, 0]


def register_config_from_final() -> RegisterConfig:
    """RegisterConfig matching ``configs/tracker/corner_fusion_register_final.py``.

    Mirrors the bridge in ``FlatSAMTracker._mask_H_register`` so the panels show
    the deployed settings rather than library defaults.

    Returns:
        The configuration used by the reference tracker.
    """
    from flatsam2.config import load_config_from_file

    tracker_cfg = load_config_from_file(
        Path(__file__).parent.parent / "configs/tracker/corner_fusion_register_final.py"
    )
    h = tracker_cfg.hough
    return RegisterConfig(
        n_angles=h.register_n_angles,
        accept_score=h.register_accept_score,
        incumbent_margin=h.register_incumbent_margin,
        appearance_power=h.register_appearance_power,
        alias_shortcut=h.register_alias_shortcut,
        sweep_stop_score=h.register_sweep_stop_score,
        sdf_roi_margin=h.register_sdf_roi_margin,
        n_boundary_points=h.align_n_boundary_points,
        max_iters=h.align_max_iters,
        huber_delta_px=h.align_huber_delta_px,
        step_tol_px=h.align_step_tol_px,
    )


def load_case(
    dataset: str,
    seq_name: str,
    frame_idx: int,
    mask_run: str,
) -> dict:
    """Load one real frame: image, SAM3 mask, template and GT corners.

    Args:
        dataset: Dataset key (e.g. ``'pot'``).
        seq_name: Sequence name.
        frame_idx: 0-based frame index of the frame to register.
        mask_run: Results directory name holding the precomputed SAM3 masks.

    Returns:
        Dict with ``init_frame``, ``init_corners``, ``init_mask``, ``frame``,
        ``mask``, ``gt_corners`` (or None) and bookkeeping fields.

    Raises:
        FileNotFoundError: When the frame or its precomputed mask is missing.
    """
    from flatsam2.registry import DATASETS

    ds = DATASETS[dataset]
    base_dir = getattr(paths, ds.base_dir_attr)
    image_paths = make_image_paths(dataset)(base_dir, seq_name)
    initfix_dir = paths.PT_INITFIX_DIR if ds.gt_loader == "pti" else None
    gt_corners, valid = make_gt_loader(dataset, initfix_dir)(base_dir, seq_name)

    init_frame = cv2.imread(str(image_paths[0]))
    frame = cv2.imread(str(image_paths[frame_idx]))
    if init_frame is None or frame is None:
        raise FileNotFoundError(f"could not read frames of {seq_name}")

    mask_dir = paths.OUTPUT_DIRECTORY / mask_run / ds.label / "masks" / seq_name
    mask_path = mask_dir / f"{frame_idx + 1:05d}.png"
    init_mask_path = mask_dir / "00001.png"
    if not mask_path.exists() or not init_mask_path.exists():
        raise FileNotFoundError(f"no precomputed mask at {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
    init_mask = cv2.imread(str(init_mask_path), cv2.IMREAD_GRAYSCALE) > 127

    return {
        "seq": seq_name,
        "frame_idx": frame_idx,
        "init_frame": init_frame,
        "init_corners": gt_corners[0],
        "init_mask": init_mask.astype(np.uint8) * 255,
        "frame": frame,
        "mask": mask,
        "gt_corners": gt_corners[frame_idx] if valid[frame_idx] else None,
    }


def _rgb(image: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float for matplotlib."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_quad(ax, corners: np.ndarray, color: str, label: Optional[str] = None,
               lw: float = 2.0, ls: str = "-", alpha: float = 1.0) -> None:
    """Plot a (2, 4) control-point quad, first corner marked."""
    ax.plot(corners[0, _QUAD_ORDER], corners[1, _QUAD_ORDER],
            color=color, lw=lw, ls=ls, alpha=alpha, label=label)
    ax.plot([corners[0, 0]], [corners[1, 0]], marker="o", ms=5, color=color, alpha=alpha)


def _mask_outline(ax, mask: np.ndarray, color: str, lw: float = 1.4,
                  label: Optional[str] = None) -> None:
    """Draw a binary mask's outer contours."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    for i, contour in enumerate(contours):
        pts = contour[:, 0, :]
        ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]),
                color=color, lw=lw, label=label if i == 0 else None)


def _crop_box(corners: np.ndarray, shape: Tuple[int, int], margin: float = 0.45):
    """Axis-aligned crop around a quad, clipped to the image."""
    x0, x1 = corners[0].min(), corners[0].max()
    y0, y1 = corners[1].min(), corners[1].max()
    pad = margin * max(x1 - x0, y1 - y0)
    return (max(0, int(x0 - pad)), min(shape[1], int(x1 + pad)),
            max(0, int(y0 - pad)), min(shape[0], int(y1 + pad)))


def _finish(ax, box) -> None:
    """Apply a crop box and strip axes."""
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[3], box[2])
    ax.set_xticks([])
    ax.set_yticks([])


def panel_inputs(case: dict, out: Path) -> None:
    """Template silhouette and the frame to be registered."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    box0 = _crop_box(case["init_corners"], case["init_frame"].shape[:2])

    axes[0].imshow(_rgb(case["init_frame"]))
    _mask_outline(axes[0], case["init_mask"] > 127, C_MASK, label="template silhouette")
    _draw_quad(axes[0], case["init_corners"], C_GT, label="template corners (GT frame 0)")
    axes[0].set_title("template: init frame, its SAM3 mask, the 4 corners", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    _finish(axes[0], box0)

    box1 = _crop_box(
        np.array(np.nonzero(case["mask"])[::-1], dtype=float), case["frame"].shape[:2], 0.25
    )
    axes[1].imshow(_rgb(case["frame"]))
    _mask_outline(axes[1], case["mask"], C_MASK, label="SAM3 mask, frame t")
    if case["gt_corners"] is not None:
        _draw_quad(axes[1], case["gt_corners"], C_GT, label="GT corners, frame t")
    axes[1].set_title(f"target: {case['seq']} frame {case['frame_idx']}", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    _finish(axes[1], box1)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def panel_sdf(case: dict, template: RegistrationTemplate, cfg: RegisterConfig,
              start_corners: np.ndarray, out: Path) -> dict:
    """The GN objective: the target SDF, the warped boundary, the residuals."""
    sdf, origin = _sdf_full(case["mask"], cfg)
    boundary_start = start_corners - origin[:, None]
    r = _residuals(sdf, template.boundary, template.corners, boundary_start)

    src = template.corners.T.astype(np.float32)
    dst = boundary_start.T.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.perspectiveTransform(
        template.boundary.reshape(-1, 1, 2).astype(np.float64), H
    ).reshape(-1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    lim = float(np.percentile(np.abs(sdf), 99))
    im = axes[0].imshow(sdf, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].scatter(warped[:, 0], warped[:, 1], s=6, c="k", label="warped template boundary")
    axes[0].set_title(
        f"signed distance field of the SAM3 mask (crop {sdf.shape[1]}x{sdf.shape[0]} px)\n"
        "sampled only at the warped boundary points", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=axes[0], fraction=0.035, label="signed distance (px)")

    axes[1].hist(r, bins=45, color=C_MASK, alpha=0.85)
    axes[1].axvline(cfg.huber_delta_px, color=C_BAD, ls="--", lw=1.2,
                    label=f"Huber δ = {cfg.huber_delta_px:g} px")
    axes[1].axvline(-cfg.huber_delta_px, color=C_BAD, ls="--", lw=1.2)
    axes[1].axvline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("residual r_i = SDF(warped boundary point)  [px]")
    axes[1].set_ylabel("boundary samples")
    axes[1].set_title(
        f"N = {len(r)} residuals; median |r| = {np.median(np.abs(r)):.2f} px", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"n_boundary": int(len(r)), "median_abs_r": float(np.median(np.abs(r)))}


def panel_gn(case: dict, template: RegistrationTemplate, cfg: RegisterConfig,
             start_corners: np.ndarray, out: Path) -> dict:
    """Control-point iterates of one damped Gauss-Newton solve.

    The iterates are recovered by re-running :func:`_gn_align` with an
    increasing iteration cap from the same start — the solver is deterministic,
    so iterate ``k`` of the capped run is iterate ``k`` of the full one.
    """
    sdf, origin = _sdf_full(case["mask"], cfg)
    start_in_field = start_corners - origin[:, None]

    iterates = [start_in_field]
    costs = []
    for k in range(0, cfg.max_iters + 1):
        control, n_iters = _gn_align(
            sdf, template.boundary, template.corners, start_in_field,
            max_iters=max(k, 1) if k > 0 else 1,
            huber_delta=cfg.huber_delta_px,
            step_tol=cfg.step_tol_px,
            trim_fraction=cfg.trim_fraction,
        ) if k > 0 else (start_in_field, 0)
        if k > 0:
            iterates.append(control)
        r = _residuals(sdf, template.boundary, template.corners, iterates[-1])
        absr = np.abs(r)
        w = np.where(absr <= cfg.huber_delta_px, 1.0,
                     cfg.huber_delta_px / np.maximum(absr, 1e-9))
        costs.append(float((w * r * r).sum()))
        if k > 0 and n_iters < k:      # solver converged before the cap
            break

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].imshow(_rgb(case["frame"]))
    _mask_outline(axes[0], case["mask"], C_MASK, lw=1.2, label="SAM3 mask")
    cmap = plt.get_cmap("viridis")
    for i, control in enumerate(iterates):
        frac = i / max(1, len(iterates) - 1)
        _draw_quad(axes[0], control + origin[:, None], cmap(frac),
                   lw=1.0 + 1.6 * frac, alpha=0.35 + 0.65 * frac,
                   label="start (perturbed)" if i == 0 else
                         (f"converged, {len(iterates) - 1} iterations" if i == len(iterates) - 1 else None))
    if case["gt_corners"] is not None:
        _draw_quad(axes[0], case["gt_corners"], C_GT, ls="--", lw=1.4, label="GT")
    axes[0].set_title("Gauss-Newton iterates over the 8 control coordinates", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    # Crop around everything the panel draws: the mask and every iterate, so a
    # solve that wanders is visible rather than cropped away.
    mask_pts = np.array(np.nonzero(case["mask"])[::-1], dtype=float)
    all_pts = np.hstack([mask_pts] + [it + origin[:, None] for it in iterates])
    _finish(axes[0], _crop_box(all_pts, case["frame"].shape[:2], 0.12))

    axes[1].plot(costs, marker="o", ms=4, color=C_MASK)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("GN iteration")
    axes[1].set_ylabel(r"$\sum_i w_i r_i^2$   (px$^2$)")
    axes[1].set_title(f"objective: {costs[0]:.3g} → {costs[-1]:.3g}", fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"n_iters": len(iterates) - 1, "cost0": costs[0], "cost1": costs[-1]}


def panel_propose(case: dict, template: RegistrationTemplate, cfg: RegisterConfig,
                  out: Path) -> dict:
    """Every second-moment proposal of the angle sweep, scored by raster overlap."""
    mask_moments = second_moments(case["mask"])
    proposals = moment_affine_inits(
        template.moments, mask_moments, template.corners,
        cfg.n_angles, cfg.scale_factors, cfg.sweep_turn_fraction,
    )
    small_shape = (max(1, int(round(case["mask"].shape[1] * template.scale))),
                   max(1, int(round(case["mask"].shape[0] * template.scale))))
    mask_small = cv2.resize(case["mask"].astype(np.uint8) * 255, small_shape,
                            interpolation=cv2.INTER_AREA) > 127
    corners_small = template.corners * template.scale

    scores, ious = [], []
    for control in proposals:
        features = _overlap(template.small_mask, mask_small, corners_small,
                            control * template.scale)
        if features is None:
            scores.append(0.0); ious.append(0.0); continue
        features["residual_px"] = 0.0
        scores.append(verification_score(features, cfg))
        ious.append(features["iou"])
    scores = np.array(scores); ious = np.array(ious)
    order = np.argsort(-scores)
    angles = np.degrees(np.linspace(0, 2 * np.pi * cfg.sweep_turn_fraction,
                                    cfg.n_angles, endpoint=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].imshow(_rgb(case["frame"]))
    _mask_outline(axes[0], case["mask"], C_MASK, lw=1.6, label="SAM3 mask")
    for i, control in enumerate(proposals):
        if i in order[: cfg.n_refine]:
            continue
        _draw_quad(axes[0], control, "0.45", lw=0.7, alpha=0.5)
    for rank, i in enumerate(order[: cfg.n_refine]):
        _draw_quad(axes[0], proposals[i], C_FUSED, lw=1.8, alpha=0.9,
                   label=f"top-{cfg.n_refine} by overlap → refined" if rank == 0 else None)
    axes[0].set_title(
        f"{len(proposals)} proposals from the second-moment sweep "
        f"({cfg.n_angles} angles)", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    _finish(axes[0], _crop_box(np.array(np.nonzero(case["mask"])[::-1], dtype=float),
                               case["frame"].shape[:2], 0.35))

    axes[1].bar(angles, scores, width=360 / max(cfg.n_angles, 1) * 0.8,
                color=[C_FUSED if i in order[: cfg.n_refine] else "0.6"
                       for i in range(len(scores))])
    axes[1].set_xlabel("sweep angle of the whitened rotation R  [deg]")
    axes[1].set_ylabel("proposal score  (IoU²)")
    axes[1].set_title("ranking that decides which proposals get refined", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"n_proposals": int(len(proposals)), "best_iou": float(ious.max())}


def panel_verify(case: dict, template: RegistrationTemplate, cfg: RegisterConfig,
                 last_H: Optional[np.ndarray], out: Path) -> dict:
    """The winner against its three cyclic relabelings: identical shape, different names."""
    result = register_mask(template, case["mask"].astype(np.uint8) * 255,
                           last_H=last_H, cfg=cfg, frame=case["frame"],
                           keep_hypotheses=True)
    if result.control is None:
        raise RuntimeError("registration produced no pose for the verify panel")

    small_shape = (max(1, int(round(case["mask"].shape[1] * template.scale))),
                   max(1, int(round(case["mask"].shape[0] * template.scale))))
    mask_small = cv2.resize(case["mask"].astype(np.uint8) * 255, small_shape,
                            interpolation=cv2.INTER_AREA) > 127
    corners_small = template.corners * template.scale
    app_scale = template.app_scale
    gray = cv2.cvtColor(case["frame"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    app_gray = cv2.resize(gray, (max(1, int(round(gray.shape[1] * app_scale))),
                                 max(1, int(round(gray.shape[0] * app_scale)))),
                          interpolation=cv2.INTER_AREA)
    app_mask = cv2.resize(case["mask"].astype(np.uint8) * 255,
                          (app_gray.shape[1], app_gray.shape[0]),
                          interpolation=cv2.INTER_AREA) > 127
    from flatsam2.mask_register import _appearance

    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    box = _crop_box(result.control, case["frame"].shape[:2], 0.35)
    for shift in range(4):
        control = np.roll(result.control, shift, axis=1)
        features = _overlap(template.small_mask, mask_small, corners_small,
                            control * template.scale) or {}
        features["residual_px"] = result.residual_px
        ncc = _appearance(template.app_gray, template.app_mask, app_gray, app_mask,
                          template.corners * app_scale, control * app_scale,
                          cfg.appearance_min_px)
        if ncc is not None:
            features["ncc"] = ncc
        score = verification_score(features, cfg)
        # Ground-truth corner error of this naming, so the panel says which
        # relabeling is actually right, not just which one the verifier likes.
        gt_err = float("nan")
        if case["gt_corners"] is not None:
            gt_err = float(np.sqrt(np.sum((control - case["gt_corners"]) ** 2) / 4.0))
        rows.append({"shift": shift, "iou": features.get("iou", float("nan")),
                     "ncc": float(ncc) if ncc is not None else float("nan"),
                     "score": score, "gt_err_px": gt_err})

        ax = axes[shift]
        ax.imshow(_rgb(case["frame"]))
        _mask_outline(ax, case["mask"], C_MASK, lw=1.2)
        _draw_quad(ax, control, C_FUSED if shift == 0 else C_BAD, lw=2.0)
        for k in range(4):
            ax.annotate(str(k), (control[0, k], control[1, k]), color="w", fontsize=9,
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15",
                                  fc=C_FUSED if shift == 0 else C_BAD, ec="none"))
        ax.set_title(
            f"{'pipeline output' if shift == 0 else f'cyclic relabel +{shift}'}\n"
            f"IoU {rows[-1]['iou']:.3f} · NCC {rows[-1]['ncc']:+.3f} · score {score:.3f}\n"
            f"corner error vs GT {rows[-1]['gt_err_px']:.1f} px",
            fontsize=9)
        _finish(ax, box)

    fig.suptitle(
        f"same silhouette, four corner namings (winning proposal source: {result.source}, "
        f"{result.n_solves} GN solves) — overlap barely separates them, texture does",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"rows": rows, "winner_source": result.source, "winner_score": float(result.score),
            "n_solves": int(result.n_solves)}


def panel_covariance(case: dict, template: RegistrationTemplate, cfg: RegisterConfig,
                     out: Path, n_boot: int = 200, seed: int = 0) -> dict:
    """Validate the analytic corner covariance against a bootstrap on real data.

    The analytic covariance of §"corner covariance" assumes the converged fit's
    residuals behave like independent noise on the boundary samples. The direct
    empirical check of that assumption is to resample the boundary samples with
    replacement, re-solve, and look at where the corners land. This panel runs
    both on the same real frame and overlays them.

    Args:
        case: Output of :func:`load_case`.
        template: Registration template context.
        cfg: Registration configuration.
        out: PNG path.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Per-corner analytic and empirical standard deviations, and their ratio.
    """
    sdf, origin = _sdf_full(case["mask"], cfg)
    # Converge from the GT pose (or the template pose when GT is missing), so
    # the panel measures the uncertainty of a *converged* fit, not of a search.
    anchor = case["gt_corners"] if case["gt_corners"] is not None else case["init_corners"]
    control, _ = _gn_align(
        sdf, template.boundary, template.corners, anchor - origin[:, None],
        max_iters=cfg.max_iters, huber_delta=cfg.huber_delta_px,
        step_tol=cfg.step_tol_px, trim_fraction=cfg.trim_fraction,
    )
    r0 = _residuals(sdf, template.boundary, template.corners, control)
    analytic = _corner_cov_from_boundary_fit(
        sdf, template.boundary, template.corners, control, r0, cfg.huber_delta_px
    )

    # The closed form assumes the residuals r_i are independent across boundary
    # samples. They are not: the samples are ordered along the contour and a
    # mask-boundary error is spatially smooth, so neighbours err together. The
    # autocorrelation gives the effective sample size that independence
    # over-counts, N_eff = N / (1 + 2 Σ_k ρ_k), and hence the variance inflation
    # the closed form is missing.
    r_c = r0 - r0.mean()
    denom = float(r_c @ r_c)
    acf = [float(r_c @ np.roll(r_c, k) / denom) for k in range(1, len(r_c) // 2)]
    tail = next((k for k, v in enumerate(acf) if v <= 0.0), len(acf))
    tau_int = 1.0 + 2.0 * float(np.sum(acf[:tail]))
    n_eff = len(r_c) / max(tau_int, 1.0)

    rng = np.random.default_rng(seed)
    n_pts = len(template.boundary)

    def resample(block: int) -> np.ndarray:
        """Bootstrap corner scatter; ``block`` = 1 is the i.i.d. resample."""
        samples = []
        for _ in range(n_boot):
            if block <= 1:
                idx = rng.integers(0, n_pts, size=n_pts)
            else:   # moving-block bootstrap: keeps runs of correlated residuals together
                starts = rng.integers(0, n_pts, size=int(np.ceil(n_pts / block)))
                idx = np.concatenate([(s + np.arange(block)) % n_pts for s in starts])[:n_pts]
            solved, _ = _gn_align(
                sdf, template.boundary[idx], template.corners, control,
                max_iters=cfg.max_iters, huber_delta=cfg.huber_delta_px,
                step_tol=cfg.step_tol_px, trim_fraction=cfg.trim_fraction,
            )
            if np.isfinite(solved).all():
                samples.append(solved)
        return np.stack(samples)

    boot = resample(1)                              # (B, 2, 4)
    block_len = max(2, int(round(tau_int)))
    boot_block = resample(block_len)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    stats = []
    for k in range(4):
        ax = axes[k]
        pts = boot[:, :, k]      # already in crop coordinates; only offsets matter
        centre = control[:, k]
        ax.scatter(pts[:, 0] - centre[0], pts[:, 1] - centre[1], s=7, alpha=0.5,
                   color=C_MASK, label=f"{len(boot)} bootstrap re-solves")
        emp = np.cov(pts, rowvar=False)
        emp_block = np.cov(boot_block[:, :, k], rowvar=False)
        for cov, color, ls, lab in ((analytic[k], C_FUSED, "-", "closed form 2σ"),
                                    (emp, C_BAD, "--", "i.i.d. bootstrap 2σ"),
                                    (emp_block, C_KP, ":", f"block bootstrap 2σ (L={block_len})")):
            vals, vecs = np.linalg.eigh(cov)     # ascending
            w = 2 * 2.0 * np.sqrt(max(vals[1], 0.0))   # major axis -> Ellipse width
            h = 2 * 2.0 * np.sqrt(max(vals[0], 0.0))
            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ax.add_patch(Ellipse((0, 0), w, h, angle=ang, fill=False,
                                 color=color, lw=1.6, ls=ls, label=lab))
        s_an = float(np.sqrt(np.trace(analytic[k]) / 2))
        s_bo = float(np.sqrt(np.trace(emp) / 2))
        s_bl = float(np.sqrt(np.trace(emp_block) / 2))
        stats.append({"corner": k, "analytic_std_px": s_an, "bootstrap_std_px": s_bo,
                      "block_bootstrap_std_px": s_bl,
                      "ratio": s_bo / max(s_an, 1e-9),
                      "block_ratio": s_bl / max(s_an, 1e-9)})
        ax.set_title(f"corner {k}\nσ: closed form {s_an:.2f} · i.i.d. {s_bo:.2f} · "
                     f"block {s_bl:.2f} px", fontsize=8.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Δx (px)")
        if k == 0:
            ax.set_ylabel("Δy (px)")
            ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle(
        f"corner uncertainty: closed form vs resampling — residual autocorrelation time "
        f"{tau_int:.1f} samples, so N_eff = {n_eff:.0f} of {len(r_c)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"per_corner": stats,
            "median_ratio": float(np.median([s["ratio"] for s in stats])),
            "median_block_ratio": float(np.median([s["block_ratio"] for s in stats])),
            "residual_autocorr_time": float(tau_int),
            "n_boundary": int(len(r_c)),
            "n_eff": float(n_eff),
            "variance_inflation_from_correlation": float(tau_int),
            "block_len": int(block_len),
            "n_boot": int(len(boot))}


def panel_jacobian(case: dict, template: RegistrationTemplate, cfg: RegisterConfig,
                   out: Path, step_px: float = 0.5, ellipse_gain: float = 25.0) -> dict:
    """Show the Jacobian that produces the covariance, on real boundary samples.

    The covariance is sigma^2 (J^T W J)^-1, and every claim about *why* one
    corner is more uncertain than another is a claim about J. This panel draws J
    directly: for each corner k, each boundary sample is coloured by how much its
    residual changes when that corner is moved one pixel,
    ``||d r_i / d c_k||``. Samples that barely respond do not constrain that
    corner, which is what an uncertainty ellipse is summarising.

    Uses exactly the finite-difference recipe of
    :func:`flatsam2.mask_align._corner_cov_from_boundary_fit` (central
    differences, same step), so what is drawn is the matrix the deployed code
    inverts, not an illustration of one.

    Args:
        case: Output of :func:`load_case`.
        template: Registration template context.
        cfg: Registration configuration.
        out: PNG path.
        step_px: Finite-difference step, matching the estimator's.
        ellipse_gain: Magnification of the drawn 1-sigma ellipse. The corner sigma
            is well under a pixel on a target hundreds of pixels wide, so an
            unmagnified ellipse is invisible; the factor is stated in the title.

    Returns:
        Per-corner sensitivity mass, effective support and sigma, for the text.
    """
    sdf, origin = _sdf_full(case["mask"], cfg)
    anchor = case["gt_corners"] if case["gt_corners"] is not None else case["init_corners"]
    control, _ = _gn_align(
        sdf, template.boundary, template.corners, anchor - origin[:, None],
        max_iters=cfg.max_iters, huber_delta=cfg.huber_delta_px,
        step_tol=cfg.step_tol_px, trim_fraction=cfg.trim_fraction,
    )
    r0 = _residuals(sdf, template.boundary, template.corners, control)
    cov = _corner_cov_from_boundary_fit(
        sdf, template.boundary, template.corners, control, r0, cfg.huber_delta_px
    )

    theta0 = corners_to_theta(control)
    jac = np.zeros((len(r0), 8))
    for p in range(8):
        plus, minus = theta0.copy(), theta0.copy()
        plus[p] += step_px
        minus[p] -= step_px
        r_plus = _residuals(sdf, template.boundary, template.corners, theta_to_corners(plus))
        r_minus = _residuals(sdf, template.boundary, template.corners, theta_to_corners(minus))
        jac[:, p] = (r_plus - r_minus) / (2.0 * step_px)

    # Sensitivity of each residual to corner k: the norm of its 2-vector of
    # partials wrt that corner's x and y.
    sens = np.stack([np.hypot(jac[:, 2 * k], jac[:, 2 * k + 1]) for k in range(4)])  # (4, N)
    H = cv2.getPerspectiveTransform(
        template.corners.T.astype(np.float32), control.T.astype(np.float32)
    )
    warped = cv2.perspectiveTransform(
        template.boundary.reshape(-1, 1, 2).astype(np.float64), H
    ).reshape(-1, 2).T + origin[:, None]
    control_img = control + origin[:, None]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    box = _crop_box(control_img, case["frame"].shape[:2], margin=0.35)
    for k, ax in enumerate(axes):
        ax.imshow(_rgb(case["frame"]))
        _mask_outline(ax, case["mask"], C_MASK, lw=1.0)
        ax.scatter(warped[0], warped[1], c=sens[k], cmap="magma", s=13,
                   vmin=0.0, vmax=float(sens.max()), zorder=3)
        ax.plot(control_img[0, _QUAD_ORDER], control_img[1, _QUAD_ORDER],
                color=C_FUSED, lw=1.2, alpha=0.85, zorder=2)
        sigma = float(np.sqrt(max(np.trace(cov[k]) / 2.0, 0.0)))
        ax.scatter([control_img[0, k]], [control_img[1, k]], marker="*", s=190,
                   color=C_GT, edgecolor="black", linewidth=0.6, zorder=4)
        eigvals, eigvecs = np.linalg.eigh(cov[k])
        ax.add_patch(Ellipse(
            (control_img[0, k], control_img[1, k]),
            width=2 * ellipse_gain * np.sqrt(max(eigvals[1], 0)),
            height=2 * ellipse_gain * np.sqrt(max(eigvals[0], 0)),
            angle=float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))),
            facecolor="none", edgecolor=C_GT, lw=1.6, zorder=4))
        share = 100.0 * float(sens[k].sum()) / float(sens.sum())
        ax.set_title(f"corner {k}:  {share:.0f}% of sensitivity,  σ ≈ {sigma:.2f} px",
                     fontsize=9)
        _finish(ax, box)
    fig.suptitle("Boundary samples coloured by |∂r/∂c_k| — how much each sample "
                 f"constrains that corner (1σ ellipse, magnified {ellipse_gain:g}x)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)

    # "Effective support": how many samples carry the bulk of the sensitivity,
    # i.e. how localized the evidence for this corner is.
    eff = [
        int(np.searchsorted(np.cumsum(np.sort(s)[::-1]) / s.sum(), 0.8) + 1)
        for s in sens
    ]
    return {
        "ellipse_gain": float(ellipse_gain),
        "n_boundary": int(len(r0)),
        "step_px": float(step_px),
        "sens_share_pct": [float(100.0 * s.sum() / sens.sum()) for s in sens],
        "eff_support": eff,
        "sigma_px": [float(np.sqrt(max(np.trace(c) / 2.0, 0.0))) for c in cov],
        "residual_rms_px": float(np.sqrt(float(r0 @ r0) / len(r0))),
        "dof": int(len(r0) - 8),
    }


def _unpack_cov(packed: np.ndarray) -> np.ndarray:
    """(4, 3) [var_x, cov_xy, var_y] rows -> (4, 2, 2) covariances."""
    out = np.zeros((4, 2, 2))
    out[:, 0, 0] = packed[:, 0]
    out[:, 0, 1] = packed[:, 1]
    out[:, 1, 0] = packed[:, 1]
    out[:, 1, 1] = packed[:, 2]
    return out


def panel_fusion(dataset: str, seq_name: str, run: str, frame_idx: int,
                 mask_run: str, out: Path, sigma: float = 2.0) -> dict:
    """Both branches' corners with covariance ellipses, and the fusion decision.

    Reads the per-frame arrays a real ``run.py`` execution wrote to
    ``<seq>_debug.npz`` — the covariances are the ones the tracker actually
    fused, not a re-derivation.
    """
    from flatsam2.registry import DATASETS
    from flatsam2.corner_covariance import mahalanobis_sq
    from flatsam2.config import load_config_from_file

    ds = DATASETS[dataset]
    npz_path = paths.OUTPUT_DIRECTORY / run / ds.label / f"{seq_name}_debug.npz"
    data = np.load(npz_path, allow_pickle=True)
    row = int(np.where(data["frame_idx"] == frame_idx)[0][0])

    case = load_case(dataset, seq_name, frame_idx, mask_run)
    init_corners = case["init_corners"]
    loma_c = warp_corners(data["loma_H"][row], init_corners)
    samh_c = warp_corners(data["samh_H"][row], init_corners)
    fused_c = warp_corners(data["fused_H"][row], init_corners)
    loma_cov = _unpack_cov(data["fusion_loma_cov"][row])
    samh_cov = _unpack_cov(data["fusion_samh_cov"][row])
    modes = str(data["fusion_modes"][row]).split(",")

    tracker_cfg = load_config_from_file(
        Path(__file__).parent.parent / f"configs/tracker/{run}.py"
    )
    gate = tracker_cfg.fusion.disagreement_gate
    # The rule inflates both covariances by measured over-confidence factors
    # before the gate; the panel must use the same numbers or it contradicts the
    # stored per-corner decisions.
    loma_scale = tracker_cfg.fusion.loma_cov_scale
    samh_scale = tracker_cfg.fusion.samh_cov_scale
    loma_cov_s = loma_cov * loma_scale
    samh_cov_s = samh_cov * samh_scale

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                            gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.imshow(_rgb(case["frame"]))
    _mask_outline(ax, case["mask"], C_MASK, lw=1.0, label="SAM3 mask")
    _draw_quad(ax, loma_c, C_KP, lw=1.4, label="keypoint branch")
    _draw_quad(ax, samh_c, C_MASK, lw=1.4, label="mask branch")
    _draw_quad(ax, fused_c, C_FUSED, lw=2.0, label="fused")
    if case["gt_corners"] is not None:
        _draw_quad(ax, case["gt_corners"], C_GT, lw=1.2, ls="--", label="GT")

    d2_raw, d2_scaled = [], []
    for k in range(4):
        for cov, centre, color in ((loma_cov_s[k], loma_c[:, k], C_KP),
                                   (samh_cov_s[k], samh_c[:, k], C_MASK)):
            # eigh returns ascending eigenvalues; Ellipse's `angle` rotates the
            # WIDTH axis, so width must be the major axis and the angle must be
            # that of the major eigenvector.
            vals, vecs = np.linalg.eigh(cov)
            width = 2 * sigma * np.sqrt(max(vals[1], 0.0))
            height = 2 * sigma * np.sqrt(max(vals[0], 0.0))
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ax.add_patch(Ellipse(centre, width, height, angle=angle,
                                 fill=False, color=color, lw=1.2, alpha=0.9))
        delta = loma_c[:, k] - samh_c[:, k]
        d2_raw.append(mahalanobis_sq(delta, loma_cov[k] + samh_cov[k]))
        d2_scaled.append(mahalanobis_sq(delta, loma_cov_s[k] + samh_cov_s[k]))
    if case["gt_corners"] is not None:
        g = case["gt_corners"]
        for k in range(4):
            ax.annotate(
                f"kp {np.linalg.norm(loma_c[:, k] - g[:, k]):.0f} / "
                f"mask {np.linalg.norm(samh_c[:, k] - g[:, k]):.0f} px",
                (g[0, k], g[1, k]), textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, color=C_GT,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.set_title(
        f"{seq_name} frame {frame_idx}: corner estimates, {sigma:g}σ ellipses of the "
        f"deployed (inflated) covariances", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _finish(ax, _crop_box(fused_c, case["frame"].shape[:2], 0.45))

    ax2 = axes[1]
    idx = np.arange(4)
    ax2.bar(idx - 0.2, d2_raw, width=0.36, color="0.62",
            label="d² with the raw fit covariances")
    ax2.bar(idx + 0.2, d2_scaled, width=0.36,
            color=[C_FUSED if m == "fused" else C_BAD for m in modes],
            label=f"d² as deployed (×{loma_scale:g} keypoint, ×{samh_scale:g} mask)")
    ax2.axhline(gate, color="k", ls="--", lw=1.2, label=f"disagreement gate = {gate:g}")
    ax2.axhline(5.99, color=C_GT, ls=":", lw=1.2,
                label="χ²₂ 95% = 5.99 (gate if calibrated)")
    ax2.set_yscale("log")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"corner {k}\n{modes[k]}" for k in range(4)], fontsize=8)
    ax2.set_ylabel("squared Mahalanobis disagreement  d²")
    ax2.set_title("per-corner decision: below the gate → precision-weighted average,\n"
                  "above → take the branch with the smaller covariance trace", fontsize=9)
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    gt_dists = {}
    if case["gt_corners"] is not None:
        g = case["gt_corners"]
        gt_dists = {
            "kp_px": [float(np.linalg.norm(loma_c[:, k] - g[:, k])) for k in range(4)],
            "mask_px": [float(np.linalg.norm(samh_c[:, k] - g[:, k])) for k in range(4)],
            "fused_px": [float(np.linalg.norm(fused_c[:, k] - g[:, k])) for k in range(4)],
        }
    return {"gt_corner_dist": gt_dists,
            "d2_raw": [float(v) for v in d2_raw],
            "d2_deployed": [float(v) for v in d2_scaled],
            "modes": modes, "gate": float(gate),
            "loma_cov_scale": float(loma_scale), "samh_cov_scale": float(samh_scale),
            "loma_trace_raw": [float(np.trace(c)) for c in loma_cov],
            "samh_trace_raw": [float(np.trace(c)) for c in samh_cov]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="pot", help="Dataset key.")
    parser.add_argument("--sequence", default="V01_1", help="Sequence name.")
    parser.add_argument("--frame", type=int, default=240, help="0-based frame index.")
    parser.add_argument("--fusion-frame", type=int, default=None,
                        help="Frame for the fusion panel (default: --frame).")
    parser.add_argument("--mask-run", default="hough_symmetry_only",
                        help="Results dir holding precomputed SAM3 masks.")
    parser.add_argument("--fusion-run", default="corner_fusion_register_final",
                        help="Results dir holding <seq>_debug.npz for the fusion panel.")
    parser.add_argument("--perturb-px", type=float, default=20.0,
                        help="Translation applied to the GN start, to show a real solve "
                             "inside the aligner's measured convergence basin.")
    parser.add_argument("--lost-shift-px", type=float, default=250.0,
                        help="Translation of the deliberately wrong incumbent pose, which "
                             "forces the stateless search instead of the fast path.")
    parser.add_argument("--out-dir", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "figs",
                        help="Directory for the PNG panels.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = register_config_from_final()
    case = load_case(args.dataset, args.sequence, args.frame, args.mask_run)
    template = RegistrationTemplate.build(
        case["init_corners"], case["init_mask"], case["frame"].shape[:2], cfg,
        init_frame=case["init_frame"],
    )
    if template is None:
        print("ERROR: could not build the registration template")
        return 1

    # A deliberately perturbed start, so the panels show a real solve rather
    # than a converged no-op: the fast path would otherwise accept immediately.
    anchor = case["gt_corners"] if case["gt_corners"] is not None else case["init_corners"]
    start = anchor + np.array([[args.perturb_px], [0.6 * args.perturb_px]])

    facts = {"case": {k: case[k] for k in ("seq", "frame_idx")},
             "config": {"n_angles": cfg.n_angles, "n_refine": cfg.n_refine,
                        "n_full": cfg.n_full, "accept_score": cfg.accept_score,
                        "incumbent_margin": cfg.incumbent_margin,
                        "huber_delta_px": cfg.huber_delta_px,
                        "appearance_power": cfg.appearance_power,
                        "alias_shortcut": cfg.alias_shortcut,
                        "sweep_stop_score": cfg.sweep_stop_score,
                        "sdf_roi_margin": cfg.sdf_roi_margin,
                        "n_boundary_points": cfg.n_boundary_points,
                        "max_iters": cfg.max_iters, "overlap_power": cfg.overlap_power},
             "perturb_px": args.perturb_px}

    panel_inputs(case, out_dir / "fig_inputs.png")
    facts["sdf"] = panel_sdf(case, template, cfg, start, out_dir / "fig_sdf.png")
    facts["gn"] = panel_gn(case, template, cfg, start, out_dir / "fig_gn.png")
    facts["propose"] = panel_propose(case, template, cfg, out_dir / "fig_propose.png")
    facts["jacobian"] = panel_jacobian(case, template, cfg, out_dir / "fig_jacobian.png")
    facts["covariance"] = panel_covariance(case, template, cfg, out_dir / "fig_covariance.png")

    # A wrong incumbent forces the stateless search, which is the path the
    # verify panel is about.
    wrong_H = cv2.getPerspectiveTransform(
        case["init_corners"].T.astype(np.float32),
        (anchor + np.array([[args.lost_shift_px], [0.0]])).T.astype(np.float32))
    facts["verify"] = panel_verify(case, template, cfg, wrong_H.astype(np.float64),
                                   out_dir / "fig_verify.png")

    fusion_frame = args.fusion_frame if args.fusion_frame is not None else args.frame
    facts["fusion"] = panel_fusion(args.dataset, args.sequence, args.fusion_run,
                                   fusion_frame, args.mask_run,
                                   out_dir / "fig_fusion.png")

    (out_dir / "figs_facts.json").write_text(json.dumps(facts, indent=2))
    print(f"Wrote panels + figs_facts.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
