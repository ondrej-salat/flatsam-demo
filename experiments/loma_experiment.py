#!/usr/bin/env python3
"""LoMa feature matching experiment for planar tracking sequences.

Computes correspondences between the init frame and every subsequent frame
using the LoMa local feature matcher.  Results are saved in two phases:

Phase 1 – Matching:
    Per-sequence NPZ file with all frame correspondences.  The file is
    written atomically at the end of a sequence so that completeness can be
    verified by loading it.

Phase 2 – Analysis:
    Per-sequence JSON file with per-frame statistics:
    target/background split, object fraction/density, and target-match
    accuracy at configurable pixel thresholds.

Usage examples:
    # POT-210, LoMaB, single sequence
    python experiments/loma_experiment.py pot --model LoMaB --sequences seq_name

    # PTI, LoMaG, full test split on SLURM subset
    python experiments/loma_experiment.py pti --model LoMaG --subset-file slurm/tmp/part1.txt

    # Re-run analysis with different thresholds (skip expensive matching)
    python experiments/loma_experiment.py pot --model LoMaB --skip-matching --thresholds 1 2 5 10
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image as _PIL_Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    corners_to_mask,
    get_sequences,
    pot_image_paths,
    pot_load_gt,
    pt_image_paths,
    pt_load_gt,
    pti_load_gt,
)
from flatsam2.geometry import warp_points
from flatsam2.path_config import paths

try:
    from loma import LoMa, LoMaB, LoMaB128, LoMaG, LoMaL, LoMaR
    _LOMA_AVAILABLE = True
except ImportError:
    _LOMA_AVAILABLE = False

_DEFAULT_OUTPUT_DIR = Path("/mnt/personal/serycjon/flatsam2/LoMa/")
_DEFAULT_THRESHOLDS = [5, 10, 15, 20]

LOMA_MODELS = {
    "LoMaB": lambda: LoMa(LoMaB()),
    "LoMaB128": lambda: LoMa(LoMaB128()),
    "LoMaL": lambda: LoMa(LoMaL()),
    "LoMaG": lambda: LoMa(LoMaG()),
    "LoMaR": lambda: LoMa(LoMaR()),
}


# ---------------------------------------------------------------------------
# Matching phase
# ---------------------------------------------------------------------------

def is_matching_complete(npz_path: Path, num_frames: int) -> bool:
    """Check whether matching output for a sequence already exists and is complete.

    Args:
        npz_path: Path to the per-sequence NPZ file.
        num_frames: Total number of frames (including the init frame at index 0).

    Returns:
        True if the NPZ exists and contains entries for every non-init frame.
    """
    try:
        data = np.load(npz_path)
        for i in range(1, num_frames):
            if f"kptsA_{i:05d}" not in data or f"kptsB_{i:05d}" not in data:
                return False
        return True
    except Exception:
        return False


def run_matching(model, image_paths: List[Path], npz_path: Path, force: bool = False) -> None:
    """Run LoMa matching between the init frame and every subsequent frame.

    Skips the sequence if matching is already complete (unless *force* is set).
    The output is written as a single NPZ file with per-frame keys:
    ``kptsA_{i:05d}`` and ``kptsB_{i:05d}`` for frame index *i*.

    Args:
        model: Loaded LoMa model instance.
        image_paths: Ordered list of image paths; index 0 is the init frame.
        npz_path: Destination NPZ file path.
        force: If True, recompute even if the file already exists.
    """
    num_frames = len(image_paths)
    if not force and is_matching_complete(npz_path, num_frames):
        return

    init_path = str(image_paths[0])
    result: dict = {}

    for i in tqdm.trange(1, num_frames, desc="  frames", leave=False, unit="frame"):
        kptsA, kptsB = model.match(init_path, str(image_paths[i]))
        result[f"kptsA_{i:05d}"] = kptsA.astype(np.float32)
        result[f"kptsB_{i:05d}"] = kptsB.astype(np.float32)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **result)


# ---------------------------------------------------------------------------
# Analysis phase
# ---------------------------------------------------------------------------

def _compute_h_gt(corners_init: np.ndarray, corners_t: np.ndarray) -> Optional[np.ndarray]:
    """Estimate GT homography from 4 corner correspondences.

    Args:
        corners_init: Init-frame corners (2, 4).
        corners_t: Current-frame corners (2, 4).

    Returns:
        3×3 homography matrix, or None if estimation fails.
    """
    src = corners_init.T.astype(np.float32)  # (4, 2)
    dst = corners_t.T.astype(np.float32)     # (4, 2)
    H, mask = cv2.findHomography(src, dst)
    if H is None or mask is None:
        return None
    return H


def run_analysis(
    npz_path: Path,
    analysis_path: Path,
    image_paths: List[Path],
    annot: np.ndarray,
    valid: np.ndarray,
    thresholds: List[int],
    data: Optional[np.lib.npyio.NpzFile] = None,
) -> List[dict]:
    """Compute and save per-frame analysis statistics for a sequence.

    Reads the pre-computed correspondences from *npz_path* and for each
    valid frame produces:
    - Object/background split counts and fractions (based on kptsA vs init mask).
    - Object/background match densities (matches per pixel).
    - Fraction of target matches within each pixel threshold of the GT
      homography prediction.

    Args:
        npz_path: Path to the per-sequence NPZ file from the matching phase.
        analysis_path: Destination JSON file path.
        image_paths: Ordered list of image paths.
        annot: GT corner annotations (num_frames, 2, 4).
        valid: Boolean valid-frame mask (num_frames,).
        thresholds: Pixel thresholds for good-match fractions.
        data: Pre-loaded NPZ data (avoids re-loading if already opened by caller).
    """
    if data is None:
        data = np.load(npz_path)

    # Determine image size from the init frame — read header only (no pixel decode).
    with _PIL_Image.open(image_paths[0]) as _img:
        W_img, H_img = _img.size

    init_mask = corners_to_mask(annot[0], (H_img, W_img))  # (H, W) uint8
    obj_area = int(init_mask.sum() // 255)
    bg_area = H_img * W_img - obj_area

    frames_out = []
    for i in range(1, len(image_paths)):
        key_A = f"kptsA_{i:05d}"
        key_B = f"kptsB_{i:05d}"
        if key_A not in data or key_B not in data:
            frames_out.append({"frame_idx": i, "valid": bool(valid[i]), "missing": True})
            continue

        kptsA = data[key_A]  # (N, 2)
        kptsB = data[key_B]  # (N, 2)
        n_matches = len(kptsA)

        # Classify kptsA as target (on object) or background.
        kptsA_int = np.clip(kptsA.round().astype(int), [0, 0], [W_img - 1, H_img - 1])
        on_obj = init_mask[kptsA_int[:, 1], kptsA_int[:, 0]] > 0  # (N,)
        n_target = int(on_obj.sum())
        n_bg = n_matches - n_target

        frame_entry: dict = {
            "frame_idx": i,
            "valid": bool(valid[i]),
            "n_matches": n_matches,
            "n_target_kpts": n_target,
            "n_bg_kpts": n_bg,
            "obj_area_px": obj_area,
            "bg_area_px": bg_area,
            "obj_fraction": n_target / n_matches if n_matches > 0 else None,
            "bg_fraction": n_bg / n_matches if n_matches > 0 else None,
            "obj_density": n_target / obj_area if obj_area > 0 else None,
            "bg_density": n_bg / bg_area if bg_area > 0 else None,
        }

        # Target-match accuracy — only for valid frames with GT corners.
        if valid[i] and n_target > 0:
            H_gt = _compute_h_gt(annot[0], annot[i])
            if H_gt is not None:
                target_kptsA = kptsA[on_obj]   # (M, 2)
                target_kptsB = kptsB[on_obj]   # (M, 2)
                proj = warp_points(H_gt, target_kptsA)  # (M, 2)
                errors = np.linalg.norm(proj - target_kptsB, axis=1)  # (M,)
                for thr in thresholds:
                    inlier_mask = errors < thr
                    frame_entry[f"target_good_at_{thr}px"] = float(inlier_mask.mean())
                    frame_entry[f"target_good_count_at_{thr}px"] = int(inlier_mask.sum())

        frames_out.append(frame_entry)

    output = {
        "thresholds_px": thresholds,
        "frames": frames_out,
    }
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(output, indent=2))
    return frames_out


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_gt_loader(
    dataset: str,
    base_dir: Path,
    initfix_dir: Optional[Path],
    woft_compat: bool = True,
) -> Callable:
    """Return the GT loader function for the given dataset variant.

    Args:
        dataset: CLI key (e.g. 'pot', 'pti').
        base_dir: Dataset base directory.
        initfix_dir: Initfix directory (only required for 'pti').
        woft_compat: POT only — if True use WOFT reannotations, if False use
            original annotations.

    Returns:
        Callable accepting (base_dir, seq_name) and returning (annot, valid).
    """
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        def _pot_loader(base_dir: Path, seq_name: str) -> Tuple[np.ndarray, np.ndarray]:
            return pot_load_gt(base_dir, seq_name, woft_compat=woft_compat)
        return _pot_loader
    if loader == "pt":
        return pt_load_gt
    if loader == "pti":
        if initfix_dir is None:
            initfix_dir = paths.PT_INITFIX_DIR
        def _pti_loader(base_dir: Path, seq_name: str) -> Tuple[np.ndarray, np.ndarray]:
            return pti_load_gt(base_dir, seq_name, initfix_dir=initfix_dir)
        return _pti_loader
    raise ValueError(f"Unknown gt_loader: {loader!r}")


def _make_image_paths_fn(dataset: str) -> Callable:
    """Return the image-paths function for the given dataset variant.

    Args:
        dataset: CLI key (e.g. 'pot', 'pti').

    Returns:
        Callable accepting (base_dir, seq_name) and returning List[Path].
    """
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_image_paths
    return pt_image_paths


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

_PCT_THRESHOLDS = [10, 20, 30, 50, 75, 90, 95]
_COUNT_THRESHOLDS = [20, 50, 100]


def _mean(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _side_length(density: Optional[float]) -> str:
    """Convert match density (matches/px²) to equivalent square side length."""
    if density is None or density <= 0:
        return "n/a"
    return f"{1.0 / density ** 0.5:.1f}px"


def _print_summary(all_frames: list[dict], thresholds: list[int], n_seq: int, n_total: int) -> None:
    """Print aggregate statistics across all analysed frames to stdout.

    Args:
        all_frames: Combined per-frame dicts from all analysed sequences.
        thresholds: Pixel thresholds used during analysis.
        n_seq: Number of sequences that were analysed.
        n_total: Total number of sequences in the dataset.
    """
    # Frames with match data (skip "missing": True entries).
    frames = [f for f in all_frames if not f.get("missing")]
    # Valid frames = have GT and match data.
    valid_frames = [f for f in frames if f.get("valid")]
    n_frames = len(frames)
    n_valid = len(valid_frames)

    if n_frames == 0:
        print("\nNo frames with match data to summarise.")
        return

    n_matches_vals = [f["n_matches"]    for f in frames if f.get("n_matches") is not None]
    obj_frac_vals  = [f["obj_fraction"] for f in frames if f.get("obj_fraction") is not None]
    bg_frac_vals   = [f["bg_fraction"]  for f in frames if f.get("bg_fraction")  is not None]
    obj_dens_vals  = [f["obj_density"]  for f in frames if f.get("obj_density")  is not None]
    bg_dens_vals   = [f["bg_density"]   for f in frames if f.get("bg_density")   is not None]

    mean_obj_dens = _mean(obj_dens_vals)
    mean_bg_dens  = _mean(bg_dens_vals)

    partial_tag = f" [PARTIAL: {n_seq}/{n_total} seq]" if n_seq < n_total else ""
    print(f"\n--- Summary{partial_tag} ---")
    print(f"  Sequences analysed : {n_seq}/{n_total}")
    print(f"  Frames             : {n_frames}  (valid with GT: {n_valid})")
    if n_matches_vals:
        print(f"  Tentative corr./frame : {_mean(n_matches_vals):.1f}  (all LoMa matches, unfiltered)")
    else:
        print(f"  Tentative corr./frame : n/a")

    print()
    print("  Match location in init frame (kptA; independent of correctness):")
    if obj_frac_vals:
        print(f"    Obj fraction  : {_mean(obj_frac_vals):.3f}  (fraction of all matches with kptA on target)")
        print(f"    BG  fraction  : {_mean(bg_frac_vals):.3f}")
    else:
        print("    Obj/BG fraction : n/a")
    if obj_dens_vals:
        print(f"    Obj density   : {mean_obj_dens:.2e} matches/px²  (~{_side_length(mean_obj_dens)} between matches)")
        print(f"    BG  density   : {mean_bg_dens:.2e} matches/px²  (~{_side_length(mean_bg_dens)} between matches)")
    else:
        print("    Obj/BG density  : n/a")

    if not thresholds:
        return

    # For accuracy metrics: use all valid frames; frames with no target kpts = 0.
    thr_cols = "".join(f"  @{t:>3d}px" for t in thresholds)
    print()
    print(f"  Target match accuracy (mean inlier fraction over {n_valid} valid frames; 0 if no target kpts):")
    print(f"    {'':10s}{thr_cols}")
    row = "    " + f"{'mean frac':10s}"
    for thr in thresholds:
        key = f"target_good_at_{thr}px"
        # All valid frames: present value or 0 (no target kpts / no GT homography).
        vals = [f.get(key, 0.0) for f in valid_frames]
        row += f"  {_mean(vals):>6.3f}" if vals else f"  {'n/a':>6s}"
    print(row)

    print()
    print(f"  Fraction of valid frames with ≥N% inliers on target ({n_valid} frames):")
    print(f"    {'N%':10s}{thr_cols}")
    for pct in _PCT_THRESHOLDS:
        row = f"    {f'≥{pct}%':10s}"
        for thr in thresholds:
            key = f"target_good_at_{thr}px"
            vals = [f.get(key, 0.0) for f in valid_frames]
            frac = float(np.mean([v >= pct / 100 for v in vals])) if vals else float("nan")
            row += f"  {frac:>6.3f}" if vals else f"  {'n/a':>6s}"
        print(row)

    print()
    print(f"  Fraction of valid frames with ≥N absolute inliers on target ({n_valid} frames):")
    print(f"    {'N':10s}{thr_cols}")
    for n_min in _COUNT_THRESHOLDS:
        row = f"    {f'≥{n_min}':10s}"
        for thr in thresholds:
            count_key = f"target_good_count_at_{thr}px"
            counts = [f.get(count_key, 0) for f in valid_frames]
            frac = float(np.mean([c >= n_min for c in counts])) if counts else float("nan")
            row += f"  {frac:>6.3f}" if counts else f"  {'n/a':>6s}"
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoMa feature matching experiment for planar tracking sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to process.")
    parser.add_argument(
        "--model",
        choices=list(LOMA_MODELS.keys()) + ["all"],
        required=True,
        help="LoMa model variant to use, or 'all' to run every variant sequentially.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Base output directory.  Results go into <output-dir>/<model>/<dataset-label>/.",
    )
    parser.add_argument("--base-dir", type=Path, help="Dataset root (defaults to path_config).")
    parser.add_argument("--initfix-dir", type=Path, help="Initfix dir for PTI (defaults to path_config).")
    parser.add_argument("--sequences", nargs="+", metavar="SEQ", help="Explicit sequence names.")
    parser.add_argument("--subset", metavar="NAME", help="Subset file name (configs/subsets/<name>.txt).")
    parser.add_argument("--subset-file", type=Path, metavar="PATH", help="Direct path to subset file (for SLURM).")
    parser.add_argument("--split", choices=["train", "test"], help="Train/test split (PT/PTI only).")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=_DEFAULT_THRESHOLDS,
        metavar="PX",
        help="Pixel thresholds for good-match fraction in analysis.",
    )
    parser.add_argument(
        "--force-matching",
        action="store_true",
        help="Recompute matching even if output already exists.",
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip the matching phase (only run analysis).",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the analysis phase (only run matching).",
    )
    parser.add_argument(
        "--no-woft-compat",
        dest="woft_compat",
        action="store_false",
        default=True,
        help="POT only: use original annotations instead of WOFT reannotations. "
             "Results go to a separate output dir (<model>/<dataset>_orig_gt/).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation after the analysis phase.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path for the HTML report.  Defaults to <output-dir>/report.html.",
    )
    return parser


def _run_single_model(args: argparse.Namespace) -> None:
    """Run matching + analysis for one model variant specified in *args.model*."""
    if not args.skip_matching and not _LOMA_AVAILABLE:
        print("ERROR: loma package not installed.  Cannot run matching phase.", file=sys.stderr)
        sys.exit(1)

    ds_info = DATASETS[args.dataset]

    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)

    if ds_info.uses_split and args.split is None:
        args.split = "test"

    # Resolve output dir: <base>/<model>/<dataset-label>/
    # For POT with original GT (woft_compat=False) use a separate dir to avoid
    # mixing analysis JSONs computed with different validity masks.
    woft_compat = getattr(args, "woft_compat", True)
    gt_suffix = "" if (DATASETS[args.dataset].gt_loader != "pot" or woft_compat) else "_orig_gt"
    output_dir = args.output_dir / args.model / (ds_info.label + gt_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect sequences.
    sequences = get_sequences(
        args.dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split,
    )

    if not sequences:
        print("No sequences found.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset   : {ds_info.label}")
    print(f"Model     : {args.model}")
    if DATASETS[args.dataset].gt_loader == "pot":
        print(f"GT annot  : {'WOFT reannotation' if woft_compat else 'original'}")
    print(f"Output    : {output_dir}")
    print(f"Sequences : {len(sequences)}")

    gt_loader = _make_gt_loader(args.dataset, args.base_dir, args.initfix_dir, woft_compat=woft_compat)
    image_paths_fn = _make_image_paths_fn(args.dataset)

    # Load LoMa model once for all sequences.
    model = None
    if not args.skip_matching:
        print(f"Loading LoMa model {args.model!r}...")
        model = LOMA_MODELS[args.model]()

    # --- Matching phase -------------------------------------------------------
    # Pre-scan so tqdm ETA/speed reflect actual matching work, not fast skips.
    if not args.skip_matching:
        needs_matching: list[tuple[str, list[Path]]] = []
        for seq_name in sequences:
            image_paths = image_paths_fn(args.base_dir, seq_name)
            npz_path = output_dir / f"{seq_name}.npz"
            if args.force_matching or not is_matching_complete(npz_path, len(image_paths)):
                needs_matching.append((seq_name, image_paths))
        n_skip = len(sequences) - len(needs_matching)
        if n_skip:
            print(f"Skipping {n_skip} already-complete sequence(s).")
        for seq_name, image_paths in tqdm.tqdm(needs_matching, desc="matching", unit="seq"):
            npz_path = output_dir / f"{seq_name}.npz"
            run_matching(model, image_paths, npz_path, force=True)

    # --- Analysis phase -------------------------------------------------------
    if not args.skip_analysis:
        n_analyzed = 0
        all_frames: list[dict] = []
        for seq_name in tqdm.tqdm(sequences, desc="analysis", unit="seq"):
            npz_path = output_dir / f"{seq_name}.npz"
            image_paths = image_paths_fn(args.base_dir, seq_name)
            # Load NPZ once; reuse for both completeness check and analysis.
            try:
                npz_data = np.load(npz_path)
            except Exception:
                continue
            num_frames = len(image_paths)
            if any(f"kptsA_{i:05d}" not in npz_data for i in range(1, num_frames)):
                continue
            analysis_path = output_dir / f"{seq_name}_analysis.json"
            annot, valid = gt_loader(args.base_dir, seq_name)
            frames = run_analysis(npz_path, analysis_path, image_paths, annot, valid, args.thresholds, data=npz_data)
            all_frames.extend(frames)
            n_analyzed += 1

        n_total = len(sequences)
        partial = n_analyzed < n_total
        if partial:
            print(
                f"\nWARNING: partial results — {n_analyzed}/{n_total} sequences analysed "
                f"({n_total - n_analyzed} missing matching output)."
            )

        _print_summary(all_frames, args.thresholds, n_analyzed, n_total)

        if not args.no_report and n_analyzed > 0:
            # Capture summary text for embedding in the HTML report.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _print_summary(all_frames, args.thresholds, n_analyzed, n_total)
            _generate_report(
                output_dir, args.model, ds_info.label, args.report_output,
                summary_text=buf.getvalue(),
            )


def _generate_report(
    output_dir: Path,
    model: str,
    dataset_label: str,
    report_output: Optional[Path],
    summary_text: str = "",
) -> None:
    """Load analysis JSONs from *output_dir* and write an interactive HTML report.

    Args:
        output_dir: Directory containing the per-sequence ``*_analysis.json`` files.
        model: LoMa model name (used in the report title).
        dataset_label: Dataset label (used in the report title).
        report_output: Explicit output path; defaults to ``output_dir/report.html``.
    """
    # Lazy import so loma_report.py is only required when generating reports.
    _experiments_dir = str(Path(__file__).resolve().parent)
    if _experiments_dir not in sys.path:
        sys.path.insert(0, _experiments_dir)

    try:
        from loma_report import load_analysis_dir, _build_html  # type: ignore[import]
    except ImportError as exc:
        print(f"WARNING: could not import loma_report — skipping HTML report ({exc})", file=sys.stderr)
        return

    try:
        report_data = load_analysis_dir(output_dir)
    except ValueError as exc:
        print(f"WARNING: report generation skipped — {exc}", file=sys.stderr)
        return

    title = f"LoMa analysis — {model} / {dataset_label}"
    html = _build_html(report_data, title, summary_text=summary_text)
    report_path = report_output or output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Report → {report_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.model == "all":
        models = list(LOMA_MODELS.keys())
        print(f"Running all models sequentially: {', '.join(models)}")
        for model_name in models:
            print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
            args.model = model_name
            _run_single_model(args)
    else:
        _run_single_model(args)


if __name__ == "__main__":
    main()
