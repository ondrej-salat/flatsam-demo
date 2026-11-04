#!/usr/bin/env python3
"""Replay corner fusion offline under alternative acceptance rules (no GPU).

Both fused branches are *decoupled* from the fused output: the SAM-H
sub-tracker runs on its own state, and LoMa's estimate depends only on the
frame, the SAM3 mask and its own keypoints (the one feedback path, SAM3
self-correction, is driven by ``loma_H``/``n_inliers``, never by the fused
homography). So the per-frame ``loma_H``/``samh_H``/covariances stored in a
run's ``{seq}_debug.npz`` are exactly what a rerun with a *different fusion
rule* would see. Changing the rule and re-fusing offline is therefore not an
approximation — it reproduces what the tracker would have output, at zero GPU
cost, for any rule that only re-weights or gates the stored branch estimates.

Rules replayed here, all against the same stored telemetry:

    baseline    — the run's own rule, read from its config (scope, naming margin,
                  covariance scales, gate). Checked against the run's eval JSON
                  at the end: it must match, or the replay does not model
                  something the tracker does and no delta here can be trusted.
    occl{R}     — the mask branch *abstains* on frames where the SAM mask is an
                  almost-perfect subset of the pose the tracker already predicts
                  (precision >= 0.90, recall <= R): the object is there and part
                  of it is hidden, so the branch's tight covariance describes a
                  precise fit to the wrong boundary. Needs ``--masks-dir``.
                  A hard switch rather than a discount — see flatsam2/occlusion.py.
    naming{M}   — the SAM-H corners are re-named from the LoMa corners when the
                  nearest cyclic rotation is M times closer than SAM-H's own
                  naming (``cfg.fusion.keypoint_naming_margin``). A silhouette
                  cannot name corners and the mask branch supplies the pose on
                  89.5% of the shipped tracker's mis-named PTi frames
                  (``experiments/naming_diagnosis.py``); this lends it the
                  keypoint branch's naming without its position.
    whole_frame — one fuse-or-select decision for all four corners instead of
                  four independent ones, so the output is always either the
                  precision-weighted average of both branches or a pose one
                  branch actually proposed (``decision_scope='whole_frame'``)
    *+covfbS    — a branch with a pose but no covariance keeps competing under
                  an isotropic sigma of S px instead of being dropped; 4.06% of
                  PTi frames discard an existing keypoint pose that way
    loma_only   — never use SAM-H (the floor any rule must beat)
    samh_only   — never use LoMa
    residual<=T — accept the SAM-H covariance only when the mask-align
                  residual that produced it is at most T px (the deployed test
                  uses ``align_residual_ok_px`` = 6.0). ``samh_accept_study``
                  finds accepted-but-wrong fits sitting at residual ~1.8 vs
                  ~0.95 for accepted-and-right ones, so this is the sharpest
                  observable separator available.
    oracle      — use SAM-H only on frames where it is actually right (error
                  <= 15 px vs GT). Not implementable online; it is the ceiling
                  on what *any* better acceptance test could buy, and says
                  whether tuning the test is worth GPU time at all.

Reported per rule: per-sequence mean p@15 (what the grid tables quote).

Imports cv2 via flatsam2.datasets — submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/fusion_rule_replay.py \\
        pti --config configs/tracker/corner_fusion_alignh_calibrated.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.config import compute_output_dir, load_config_from_file
from flatsam2.datasets import pot_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error, best_rotation_error
from flatsam2.occlusion import is_occluded_view, warp_template_silhouette
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences
from flatsam2.tracking.corner_fusion import fuse_corners

_P_AT_K_PX = 15.0
_ORACLE_OK_PX = 15.0
_RESIDUAL_THRESHOLDS = (1.0, 1.5, 2.0, 3.0, 4.0)


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    raise ValueError(f"fusion_rule_replay supports pot/pti GT loaders, got {loader!r}")


def unpack_cov(packed: np.ndarray) -> np.ndarray | None:
    """(4, 3) packed (xx, xy, yy) per corner -> (4, 2, 2), or None if unusable.

    Args:
        packed: (4, 3) covariance row for one frame; NaN when the branch had no
            covariance that frame.

    Returns:
        (4, 2, 2) symmetric per-corner covariance, or None when any entry is
        not finite (the fuser's "no opinion" case).
    """
    if not np.all(np.isfinite(packed)):
        return None
    cov = np.zeros((4, 2, 2))
    cov[:, 0, 0] = packed[:, 0]
    cov[:, 0, 1] = packed[:, 1]
    cov[:, 1, 0] = packed[:, 1]
    cov[:, 1, 1] = packed[:, 2]
    return cov


def _finite_H(H: np.ndarray) -> np.ndarray | None:
    """The homography, or None when the run recorded it as absent (NaN)."""
    return H if np.all(np.isfinite(H)) else None


def _h_from_corners(init_corners: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """Exact 4-point homography (2, 4) -> (2, 4), or None when degenerate."""
    try:
        H = cv2.getPerspectiveTransform(
            init_corners.T.astype(np.float32), corners.T.astype(np.float32)
        )
    except cv2.error:
        return None
    return H.astype(np.float64) if np.isfinite(H).all() else None


def _load_mask(masks_dir: Path, seq_name: str, frame_idx: int) -> np.ndarray | None:
    """Selected SAM3 mask for one frame from the candidate dump, or None.

    The dump names frames 1-based (``{frame+1:05d}.png``), which is why this
    exists rather than an inline path join.
    """
    path = masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def replay_sequence(
    data,
    init_corners: np.ndarray,
    gt_corners: np.ndarray,
    valid: np.ndarray,
    rule: str,
    loma_cov_scale: float,
    samh_cov_scale: float,
    disagreement_gate: float,
    residual_max: float,
    decision_scope: str = "per_corner",
    missing_cov_px: float = 0.0,
    keypoint_naming_margin: float = 0.0,
    occlusion_max_recall: float = 0.0,
    masks_dir: Path | None = None,
    seq_name: str = "",
    template_mask: np.ndarray | None = None,
) -> tuple[float, int, int, int]:
    """Re-fuse one sequence's stored branch estimates under ``rule``.

    Args:
        data: Opened ``{seq}_debug.npz``.
        init_corners: (2, 4) template corners.
        gt_corners: (N, 2, 4) GT corners.
        valid: (N,) GT-availability flags.
        rule: One of ``baseline``, ``loma_only``, ``samh_only``, ``residual``,
            ``oracle``.
        loma_cov_scale: Calibration scale for the LoMa covariance.
        samh_cov_scale: Same, SAM-H.
        disagreement_gate: Squared-Mahalanobis fuse/arbitrate boundary.
        residual_max: Acceptance threshold for the ``residual`` rule (px).
        decision_scope: ``per_corner`` or ``whole_frame`` (see
            ``flatsam2.tracking.corner_fusion.fuse_corners``).
        missing_cov_px: When positive, a branch that produced a pose but no
            covariance is given an isotropic covariance of this standard
            deviation instead of being dropped. 4.06% of PTi frames have a
            keypoint pose discarded for want of a covariance
            (``experiments/corner_complementarity.py``); this measures whether
            letting it compete under a stated-wide prior is worth anything.
        keypoint_naming_margin: Re-name the SAM-H corners from the LoMa corners
            when the nearest cyclic rotation is that many times closer than
            SAM-H's own naming (``cfg.fusion.keypoint_naming_margin``). A pure
            re-labelling of a stored estimate, so it stays inside what this
            replay reproduces exactly.

    Returns:
        Tuple ``(p@15, n_ok, n_gt, n_br_ok)`` for this sequence; ``p@15`` is NaN
        when the sequence has no GT frames. ``n_br_ok`` counts frames within the
        threshold under the *best* cyclic corner rotation, so ``n_br_ok - n_ok``
        is the naming error and the rest is localization.
    """
    frame_idx = data["frame_idx"]
    loma_H_all = data["loma_H"]
    samh_H_all = data["samh_H"]
    loma_cov_all = data["fusion_loma_cov"]
    samh_cov_all = data["fusion_samh_cov"]
    residual_all = data["maskh_align_residual_px"]

    last_corners = gt_corners[0].copy()
    n_ok = 0
    n_gt = 0
    n_br_ok = 0

    for t, fidx in enumerate(frame_idx):
        loma_H = _finite_H(loma_H_all[t])
        samh_H = _finite_H(samh_H_all[t])
        loma_cov = unpack_cov(loma_cov_all[t])
        samh_cov = unpack_cov(samh_cov_all[t])

        if occlusion_max_recall > 0.0 and masks_dir is not None and samh_H is not None:
            # Predict this frame's silhouette by warping the template's *own*
            # first-frame mask (quad-free: the object need not be a quad), from
            # the keypoint branch's current-frame pose when it has one. LoMa is
            # a current-frame estimate, so unlike last frame's pose it does not
            # turn fast motion into false occlusion; and using it to judge the
            # *mask* branch is not circular, since it never judges itself.
            predict_H = loma_H if loma_H is not None else _h_from_corners(
                init_corners, last_corners
            )
            mask = _load_mask(masks_dir, seq_name, int(fidx))
            if mask is not None and predict_H is not None and template_mask is not None:
                predicted = warp_template_silhouette(
                    template_mask, predict_H, mask.shape
                )
                if is_occluded_view(mask, predicted, max_recall=occlusion_max_recall):
                    samh_H, samh_cov = None, None

        if rule == "loma_only":
            samh_H, samh_cov = None, None
        elif rule == "samh_only":
            loma_H, loma_cov = None, None
        elif rule == "residual":
            if not (residual_all[t] <= residual_max):
                samh_cov = None
        elif rule == "oracle":
            samh_err = (
                alignment_error(gt_corners[fidx], warp_corners(samh_H, init_corners))
                if samh_H is not None and fidx < len(valid) and valid[fidx]
                else float("inf")
            )
            if not samh_err <= _ORACLE_OK_PX:
                samh_cov = None

        if missing_cov_px > 0.0:
            fallback = np.tile(np.eye(2) * missing_cov_px**2, (4, 1, 1))
            if loma_H is not None and loma_cov is None:
                loma_cov = fallback / loma_cov_scale
            if samh_H is not None and samh_cov is None:
                samh_cov = fallback / samh_cov_scale

        fused, _ = fuse_corners(
            init_corners,
            loma_H,
            loma_cov,
            samh_H,
            samh_cov,
            last_corners,
            disagreement_gate=disagreement_gate,
            loma_cov_scale=loma_cov_scale,
            samh_cov_scale=samh_cov_scale,
            decision_scope=decision_scope,
            keypoint_naming_margin=keypoint_naming_margin,
        )
        last_corners = fused

        if fidx < len(valid) and valid[fidx]:
            n_gt += 1
            if alignment_error(gt_corners[fidx], fused) <= _P_AT_K_PX:
                n_ok += 1
            if best_rotation_error(gt_corners[fidx], fused)[0] <= _P_AT_K_PX:
                n_br_ok += 1

    p_at_k = 100.0 * n_ok / n_gt if n_gt else float("nan")
    return p_at_k, n_ok, n_gt, n_br_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    # Default to the *config's own* fusion settings. Hard-coded defaults made
    # the row labelled `baseline` something the run never did: a whole-frame run
    # was compared against a per-corner replay, so every delta was measured from
    # the wrong reference (review, 2026-08-19).
    parser.add_argument("--loma-cov-scale", type=float, default=None)
    parser.add_argument("--samh-cov-scale", type=float, default=None)
    parser.add_argument("--disagreement-gate", type=float, default=None)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write the per-rule pooled/mean p@15 here, for the docs generator.",
    )
    parser.add_argument(
        "--missing-cov-px",
        type=float,
        nargs="*",
        default=[10.0],
        help=(
            "Isotropic sigma (px) given to a branch that has a pose but no covariance, "
            "instead of dropping it. Empty list disables those rows."
        ),
    )
    parser.add_argument(
        "--naming-margins",
        type=float,
        nargs="*",
        default=[1.0, 1.5, 2.0, 3.0],
        help=(
            "Re-name the SAM-H corners from the LoMa corners when the nearest "
            "cyclic rotation is this many times closer than SAM-H's own naming. "
            "Empty list disables those rows."
        ),
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help=(
            "sam3_candidate_dump masks root for this dataset "
            "(<results>/sam3_candidates/<label>/masks). Enables the occl{R} rules, "
            "which drop the mask branch on frames where the mask is a clean subset "
            "of the predicted silhouette."
        ),
    )
    parser.add_argument(
        "--occlusion-recalls", type=float, nargs="*", default=[0.7, 0.8, 0.9],
        help="Recall thresholds below which the mask counts as a partial view.",
    )
    args = parser.parse_args(argv)

    tracker_cfg = load_config_from_file(args.config)
    fusion_cfg = tracker_cfg.fusion
    if args.loma_cov_scale is None:
        args.loma_cov_scale = fusion_cfg.loma_cov_scale
    if args.samh_cov_scale is None:
        args.samh_cov_scale = fusion_cfg.samh_cov_scale
    if args.disagreement_gate is None:
        args.disagreement_gate = fusion_cfg.disagreement_gate
    base_scope = fusion_cfg.decision_scope
    base_naming = fusion_cfg.keypoint_naming_margin

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)
    sequences = get_sequences(
        args.dataset, base_dir, sequences=args.sequences, subset=args.subset, split=args.split
    )
    load_gt_fn = _load_gt_fn(args.dataset)

    # (label, rule, residual threshold, decision scope, missing-covariance sigma,
    #  keypoint-naming margin, occlusion max-recall)
    rules: list[tuple[str, str, float, str, float, float, float]] = [
        ("baseline", "baseline", 0.0, base_scope, 0.0, base_naming, 0.0),
        ("whole_frame", "baseline", 0.0, "whole_frame", 0.0, 0.0, 0.0),
        *[
            (f"whole_frame+covfb{s:g}", "baseline", 0.0, "whole_frame", s, 0.0, 0.0)
            for s in args.missing_cov_px
        ],
        *[
            (f"per_corner+covfb{s:g}", "baseline", 0.0, "per_corner", s, 0.0, 0.0)
            for s in args.missing_cov_px
        ],
        *[
            (f"whole_frame+naming{m:g}", "baseline", 0.0, "whole_frame", 0.0, m, 0.0)
            for m in args.naming_margins
        ],
        ("loma_only", "loma_only", 0.0, "per_corner", 0.0, 0.0, 0.0),
        ("samh_only", "samh_only", 0.0, "per_corner", 0.0, 0.0, 0.0),
        *[(f"residual<={t}", "residual", t, "per_corner", 0.0, 0.0, 0.0)
          for t in _RESIDUAL_THRESHOLDS],
        ("oracle", "oracle", 0.0, "per_corner", 0.0, 0.0, 0.0),
        ("oracle+whole_frame", "oracle", 0.0, "whole_frame", 0.0, 0.0, 0.0),
        *[
            (f"whole_frame+occl{r:g}", "baseline", 0.0, "whole_frame", 0.0, 0.0, r)
            for r in (args.occlusion_recalls if args.masks_dir else [])
        ],
        *[
            (f"whole_frame+naming2+occl{r:g}", "baseline", 0.0, "whole_frame", 0.0, 2.0, r)
            for r in (args.occlusion_recalls if args.masks_dir else [])
        ],
    ]
    scores: dict[str, list[float]] = {label: [] for label, *_ in rules}
    pooled: dict[str, list[tuple[int, int]]] = {label: [] for label, *_ in rules}
    pooled_br: dict[str, int] = {label: 0 for label, *_ in rules}

    print(
        f"Fusion-rule replay: config={args.config.stem}  dataset={ds_info.label}  "
        f"n_sequences={len(sequences)}  scales loma={args.loma_cov_scale} "
        f"samh={args.samh_cov_scale}  gate={args.disagreement_gate}"
    )

    n_seqs = 0
    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            if "fusion_samh_cov" not in data:
                continue
            n_seqs += 1
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            init_corners = gt_corners[0]
            # The template silhouette is the first frame's segmentation, not a
            # quad through its corners — that is what "quad-free" means here.
            template_mask = (
                _load_mask(args.masks_dir, seq_name, 0) if args.masks_dir else None
            )
            for label, rule, threshold, scope, miss_px, naming_margin, occl_r in rules:
                p_at_k, n_ok, n_gt, n_br_ok = replay_sequence(
                    data,
                    init_corners,
                    gt_corners,
                    valid,
                    rule,
                    args.loma_cov_scale,
                    args.samh_cov_scale,
                    args.disagreement_gate,
                    threshold,
                    decision_scope=scope,
                    missing_cov_px=miss_px,
                    keypoint_naming_margin=naming_margin,
                    occlusion_max_recall=occl_r,
                    masks_dir=args.masks_dir,
                    seq_name=seq_name,
                    template_mask=template_mask,
                )
                scores[label].append(p_at_k)
                pooled[label].append((n_ok, n_gt))
                pooled_br[label] += n_br_ok

    if n_seqs == 0:
        print("No corner_fusion debug npz found — nothing to replay.")
        return 1

    total_gt = sum(n_gt for _, n_gt in pooled["baseline"])
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "config": args.config.stem,
            "dataset": ds_info.label,
            "n_sequences": n_seqs,
            "total_gt_frames": total_gt,
            "gate": args.disagreement_gate,
            "loma_cov_scale": args.loma_cov_scale,
            "samh_cov_scale": args.samh_cov_scale,
            "rules": {
                label: {
                    "pooled_p15": 100.0 * sum(o for o, _ in pooled[label]) / total_gt
                    if total_gt else float("nan"),
                    "mean_p15": float(np.nanmean(scores[label])),
                    "pooled_br_p15": 100.0 * pooled_br[label] / total_gt
                    if total_gt else float("nan"),
                }
                for label, *_ in rules
            },
        }, indent=2))
        print(f"Wrote {args.out_json}")
    print(f"\nReplayed {n_seqs} sequences, {total_gt} GT frames.\n")

    # The whole instrument rests on the claim that re-fusing stored branch
    # estimates reproduces what the tracker emitted. Check it here rather than
    # asserting it in prose: the baseline row replays the config's own rule, so
    # it must match the run's eval. A mismatch means some part of the rule is
    # not modelled and every delta below is suspect.
    eval_path = seq_dir / "eval_results.json"
    if eval_path.exists() and total_gt:
        stored = json.loads(eval_path.read_text())["aggregate"]
        replayed = 100.0 * sum(o for o, _ in pooled["baseline"]) / total_gt
        delta = replayed - stored.get("overall_p@15", float("nan"))
        verdict = "MATCH" if abs(delta) < 0.005 else "MISMATCH -- deltas below are suspect"
        print(
            f"  self-check: baseline replay {replayed:.2f} vs run eval "
            f"{stored.get('overall_p@15', float('nan')):.2f}  ({delta:+.3f})  {verdict}\n"
        )
    else:
        print("  self-check: no eval_results.json to compare the baseline against\n")
    # Pooled first (every GT frame counted once, the benchmark's own headline),
    # per-sequence mean second (what the grid tables quote).
    print("  rule                  | pooled p@15 | pooled br | br-gap | mean p@15 | d vs base")
    print("  " + "-" * 78)
    base_pooled = 100.0 * sum(o for o, _ in pooled["baseline"]) / total_gt if total_gt else 0.0
    for label, *_ in rules:
        n_ok = sum(o for o, _ in pooled[label])
        pooled_p = 100.0 * n_ok / total_gt if total_gt else float("nan")
        br_p = 100.0 * pooled_br[label] / total_gt if total_gt else float("nan")
        print(
            f"  {label:<21} | {pooled_p:11.2f} | {br_p:9.2f} | {br_p - pooled_p:6.2f} | "
            f"{np.nanmean(scores[label]):9.2f} | {pooled_p - base_pooled:+9.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
