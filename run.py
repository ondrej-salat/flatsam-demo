"""Run tracking experiments on POT-210 and PlanarTrack datasets.

This script runs a tracker on dataset sequences and saves results in the
standard format for evaluation.
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from einops import rearrange
from tqdm import tqdm

from flatsam2 import match_naming
from flatsam2.config import TrackerConfig, compute_output_dir, load_config_from_file
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    make_gt_loader,
    make_image_paths,
    warp_corners,
)
from flatsam2.path_config import paths
from flatsam2.reproducibility import save_run_info
from flatsam2.tracker import _DEBUG_CANDIDATES, FlatSAMTracker, TrackResult
from flatsam2.tracking.common import CorrectionTelemetry, MaskHTelemetry


def _attach_persistent_error_log(output_dir: Path) -> Path:
    """Persist tracker WARNING/ERROR logs into the output dir (permanent storage).

    ``step_with_mask`` swallows per-frame exceptions and only ``logger.warning``s
    them, so console output is easy to lose. Writing them next to the results
    guarantees a swallowed failure (e.g. a SAM3 correction that crashed after
    mutating memory) is investigable later.

    Args:
        output_dir: Per-config results dir; the log goes to ``.repro/``.

    Returns:
        Path to the log file.
    """
    repro_dir = output_dir / ".repro"
    repro_dir.mkdir(parents=True, exist_ok=True)
    log_path = repro_dir / "track_errors.log"

    flatsam2_logger = logging.getLogger("flatsam2")
    for existing in flatsam2_logger.handlers:
        if getattr(existing, "_flatsam2_track_error_handler", False):
            return log_path  # already attached (run_experiments called twice)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler._flatsam2_track_error_handler = True  # type: ignore[attr-defined]
    flatsam2_logger.addHandler(handler)
    if flatsam2_logger.level == logging.NOTSET or flatsam2_logger.level > logging.WARNING:
        flatsam2_logger.setLevel(logging.WARNING)
    return log_path


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def run_sequence(
    tracker: FlatSAMTracker,
    image_paths: list,
    init_corners: np.ndarray,
    seq_name: str,
    output_dir: Path,
    save_masks: bool = False,
) -> np.ndarray:
    num_frames = len(image_paths)
    all_corners = np.zeros((num_frames, 2, 4))
    debug_records: list = []  # one TrackerDebugInfo per non-init frame (None on error)
    debug_frame_ids: list = []  # 0-based frame index corresponding to each record
    if save_masks:
        seq_output_dir = output_dir / "masks" / seq_name
        seq_output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, img_path in enumerate(
        tqdm(image_paths, desc=f"Tracking {seq_name}", disable=True)
    ):
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Failed to load image: {img_path}")
        if frame_idx == 0:
            tracker.init(frame, init_corners=init_corners)
            all_corners[frame_idx] = init_corners
            if save_masks:
                mask_path = seq_output_dir / f"{frame_idx + 1:05d}.png"
                cv2.imwrite(str(mask_path), tracker.state.init_mask)
        else:
            result: TrackResult = tracker.track(frame)
            current_corners = warp_corners(result.homography, init_corners)
            all_corners[frame_idx] = current_corners
            debug_records.append(result.debug)
            debug_frame_ids.append(frame_idx)
            if save_masks:
                mask_path = seq_output_dir / f"{frame_idx + 1:05d}.png"
                cv2.imwrite(str(mask_path), np.uint8(result.mask) * 255)
                # When a correction fired, also save the injected mask alongside
                # the original (suffix _corr) so both are available for viewing.
                if result.correction_mask is not None:
                    corr_path = seq_output_dir / f"{frame_idx + 1:05d}_corr.png"
                    cv2.imwrite(str(corr_path), np.uint8(result.correction_mask) * 255)
    _save_debug(debug_records, debug_frame_ids, output_dir, seq_name)
    return all_corners


_NAN_H = np.full((3, 3), np.nan)


def _save_debug(
    debug_records: list,
    frame_ids: list,
    output_dir: Path,
    seq_name: str,
) -> None:
    """Save per-frame tracker telemetry to ``{seq_name}_debug.npz``.

    Written for every run: it is the single sidecar capturing what each frame
    did, for visualization and debugging. No-op only when there are no tracked
    frames.

    The npz holds arrays of length T (tracked frames = total minus the init
    frame). For each candidate in :data:`flatsam2.tracker._DEBUG_CANDIDATES`
    (``flow_H``, ``redetection_flow_H``, ``mask_H``, ``loma_H``, ``samh_H``):
        <cand>        — (T, 3, 3), NaN when the candidate was not computed
    plus:
        frame_idx — (T,) int32, 0-based frame index in the sequence
        selected  — (T,) str, candidate actually used (e.g. 'loma_H', 'samh_H')
        corrected — (T,) bool, whether a SAM3 self-correction fired this frame
        n_inliers — (T,) int32, LoMa RANSAC inlier count (-1 when N/A)
    and, for correction-enabled runs (NaN/False otherwise), the always-on
    correction observables from :class:`CorrectionTelemetry`:
        corr_spill_loma / corr_excess_loma / corr_area_ratio_loma — (T,) float32,
            superset-guard value / excess-depth dout / area ratio of the SAM mask
            vs the LoMa-induced mask
        corr_trigger — (T,) bool, trigger fired (corr_trigger & ~corrected =
            abstain or dry run)
    and the mask-H observables from :class:`MaskHTelemetry`:
        maskh_source — (T,) str, 'hough' when the Hough quad produced mask-H,
            'none' when it was unavailable
        naming_support — (T, 4) float32, correspondence support (effective
            matches) of the four cyclic namings of the mask branch's pose.
        naming_n_matches — (T,) int32, raw match count behind it (-1 if N/A).
        naming_shift — (T,) int32, cyclic shift the namer applied (0 = kept the
            incumbent, 1-3 = re-named, -1 = did not run). Distinguishes "did not
            help" from "never fired".
        naming_hist_counts / naming_hist_sumsq — (T, 4, B) residual histogram
            per naming and the per-bin sum of squared residuals, over
            naming_hist_edges. Makes an offline re-test of the support's tau
            exact rather than bin-resolution.

    Oracle selection can be computed offline by warping init_corners with each
    stored H and comparing to GT corners.
    """
    if not debug_records:
        return

    arrays: dict[str, np.ndarray] = {}
    arrays["frame_idx"] = np.array(frame_ids, dtype=np.int32)

    for candidate_name in _DEBUG_CANDIDATES:
        H_list = []
        for rec in debug_records:
            cand = getattr(rec, candidate_name) if rec is not None else None
            H_list.append(_NAN_H.copy() if cand is None or cand.H is None else cand.H)
        arrays[candidate_name] = np.stack(H_list)  # (T, 3, 3)

    arrays["selected"] = np.array(
        [rec.selected if rec is not None else "" for rec in debug_records]
    )
    arrays["corrected"] = np.array(
        [bool(rec.corrected) if rec is not None else False for rec in debug_records],
        dtype=bool,
    )
    arrays["n_inliers"] = np.array(
        [rec.n_inliers if rec is not None else -1 for rec in debug_records],
        dtype=np.int32,
    )

    correction_records = [
        rec.correction if rec is not None else CorrectionTelemetry()
        for rec in debug_records
    ]
    _CORR_FLOAT_FIELDS = (
        "spill_loma",
        "excess_loma",
        "area_ratio_loma",
    )
    for field_name in _CORR_FLOAT_FIELDS:
        arrays[f"corr_{field_name}"] = np.array(
            [getattr(tel, field_name) for tel in correction_records],
            dtype=np.float32,
        )
    arrays["corr_trigger"] = np.array(
        [bool(tel.trigger_fired) for tel in correction_records],
        dtype=bool,
    )

    # Whether the Hough quad produced mask-H each frame.
    maskh_records = [
        rec.mask_h if rec is not None else MaskHTelemetry() for rec in debug_records
    ]
    arrays["maskh_source"] = np.array([tel.source for tel in maskh_records])

    # Corner-naming evidence: the correspondence support of the four cyclic
    # namings of each frame's mask-branch pose, plus the residual histogram
    # behind it. Written so an alternative naming rule (another tau, a margin
    # over the incumbent, an abstain threshold) can be scored against a
    # finished run offline instead of costing a grid per question.
    n_bins = match_naming.N_RESIDUAL_BINS
    naming_support = np.full((len(maskh_records), 4), np.nan, dtype=np.float32)
    naming_counts = np.zeros((len(maskh_records), 4, n_bins), dtype=np.int32)
    naming_sumsq = np.zeros((len(maskh_records), 4, n_bins), dtype=np.float32)
    for t, tel in enumerate(maskh_records):
        if tel.naming_support is not None:
            naming_support[t] = tel.naming_support
        if tel.naming_hist_counts is not None:
            naming_counts[t] = tel.naming_hist_counts
        if tel.naming_hist_sumsq is not None:
            naming_sumsq[t] = tel.naming_hist_sumsq
    arrays["naming_support"] = naming_support
    arrays["naming_n_matches"] = np.array(
        [tel.naming_n_matches for tel in maskh_records], dtype=np.int32
    )
    arrays["naming_shift"] = np.array(
        [tel.naming_shift for tel in maskh_records], dtype=np.int32
    )
    arrays["naming_hist_counts"] = naming_counts
    arrays["naming_hist_sumsq"] = naming_sumsq
    arrays["naming_hist_edges"] = np.array(
        match_naming.RESIDUAL_BIN_EDGES, dtype=np.float32
    )

    out_path = output_dir / f"{seq_name}_debug.npz"
    np.savez(out_path, **arrays)
    print(f"Saved debug info to: {out_path}")


def save_corners(corners: np.ndarray, output_path: Path) -> None:
    corners_flat = rearrange(
        corners, "frames xy corners -> frames (corners xy)", corners=4, xy=2
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, corners_flat, fmt="%.6f")


def is_sequence_complete(output_dir: Path, seq_name: str, num_frames: int) -> bool:
    corner_file = output_dir / f"{seq_name}.txt"
    if not corner_file.exists():
        return False
    try:
        corners = np.loadtxt(corner_file)
        if corners.ndim == 1:
            corners = corners[np.newaxis, :]
        return len(corners) == num_frames
    except Exception:
        return False


def run_experiments(
    dataset: str,
    base_dir: Path,
    base_output_dir: Path,
    sequences: list,
    split: Optional[str] = None,
    save_masks: bool = False,
    skip_completed: bool = True,
    config: Optional[TrackerConfig] = None,
    config_path: Optional[Path] = None,
    fail_hard: bool = False,
    initfix_dir: Optional[Path] = None,
) -> None:
    ds_info = DATASETS[dataset]
    load_gt_fn = make_gt_loader(dataset, initfix_dir)
    img_paths_fn = make_image_paths(dataset)

    output_dir = compute_output_dir(base_output_dir, dataset, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = _attach_persistent_error_log(output_dir)
    print(f"Tracker warnings/errors logged to: {error_log_path}")
    save_run_info(output_dir, config=config, config_path=config_path)

    # Correction rewrites the SAM3 mask stream, so the masks actually used during
    # tracking diverge from any shared precomputed set. Persist them per-config so
    # the detail viewer shows the real masks (and their corrections), never a
    # recomputed/shared fallback.
    if config is not None and config.correction.enabled and not save_masks:
        print("Correction enabled: forcing mask saving (masks reflect SAM3 corrections).")
        save_masks = True

    print(f"Processing {len(sequences)} {ds_info.label} sequences")
    if split:
        print(f"Split: {split}")
    print(f"Output directory: {output_dir}")
    if config:
        flow_desc = (
            f"{config.flow.model}/{config.flow.weights}"
            if config.flow_enabled
            else "disabled"
        )
        print(
            f"Config: algorithm={config.algorithm}, flow={flow_desc}, "
            f"segmenter={config.segmenter.model}"
        )

    tracker = FlatSAMTracker(config, fail_hard=fail_hard)
    start_time = time.time()
    processed_count = 0
    total_seqs = len(sequences)
    skipped_count = 0
    # Sequences that raised and were skipped. A run that swallows these and then
    # exits 0 would let a chained evaluation aggregate whatever happens to be on
    # disk. The list is returned so the caller can exit non-zero instead.
    failed_sequences: list[str] = []
    total_tracked_time = 0.0

    for seq_name in sequences:
        print(f"\n{'=' * 60}")
        print(f"Sequence: {seq_name}")
        print(f"{'=' * 60}")

        image_paths = img_paths_fn(base_dir, seq_name)
        if skip_completed and is_sequence_complete(output_dir, seq_name, len(image_paths)):
            print(f"Skipping {seq_name} (already complete)")
            skipped_count += 1
            tracked_remaining = (total_seqs - skipped_count) - processed_count
            total_elapsed = time.time() - start_time
            avg = total_tracked_time / processed_count if processed_count > 0 else 0.0
            eta = avg * tracked_remaining
            if processed_count > 0:
                print(
                    f"Elapsed: {format_time(total_elapsed)} | ETA: {format_time(eta)} (tracked only)"
                )
            else:
                print(
                    f"Elapsed: {format_time(total_elapsed)} | ETA: N/A (tracked only)"
                )
            continue

        seq_start = time.time()
        num_frames = 0
        try:
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            image_paths = img_paths_fn(base_dir, seq_name)
            num_frames = len(image_paths)
            init_corners = gt_corners[0]

            pred_corners = run_sequence(
                tracker,
                image_paths,
                init_corners,
                seq_name,
                output_dir,
                save_masks,
            )

            output_file = output_dir / f"{seq_name}.txt"
            save_corners(pred_corners, output_file)
            print(f"Saved results to: {output_file}")
        except Exception as e:
            if fail_hard:
                raise
            failed_sequences.append(seq_name)
            print(f"ERROR processing {seq_name}: {e}")
            traceback.print_exc()
        finally:
            seq_elapsed = time.time() - seq_start
            processed_count += 1
            total_tracked_time += seq_elapsed
            total_elapsed = time.time() - start_time
            avg = total_tracked_time / processed_count if processed_count else 0.0
            tracked_remaining = (total_seqs - skipped_count) - processed_count
            eta = avg * tracked_remaining
            fps = num_frames / seq_elapsed if seq_elapsed > 0 and num_frames else 0.0
            print(
                f"Sequence time: {format_time(seq_elapsed)} | FPS: {fps:.2f} | "
                f"Total elapsed: {format_time(total_elapsed)} | "
                f"ETA: {format_time(eta)} (tracked only)"
            )

    if failed_sequences:
        print(
            f"\n{len(failed_sequences)} sequence(s) FAILED: "
            + ", ".join(failed_sequences[:10])
            + (" ..." if len(failed_sequences) > 10 else "")
        )
    return failed_sequences


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run tracking experiments on POT-210 or PlanarTrack datasets"
    )
    parser.add_argument(
        "dataset", choices=CLI_CHOICES, help="Dataset to run experiments on"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Dataset base directory (uses path_config.py default if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(paths.OUTPUT_DIRECTORY),
        help="Top-level output directory (subdirectories will be created based on config and dataset)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequences to process (if not specified, process all)",
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        help=(
            "Path to a subset file (one sequence name per line). Combined with "
            "--sequences, the intersection is used."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Dataset split (train or test) — only applicable for datasets that support splits",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-sequence initfix files for planartrack-initfix / pti "
            "(defaults to PT_INITFIX_DIR from path_config.py)"
        ),
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save mask images in addition to corner coordinates",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Re-process sequences even if they are already complete",
    )
    parser.add_argument(
        "--fail-hard",
        action="store_true",
        help=(
            "Propagate exceptions from the tracker instead of suppressing them. "
            "Use this to surface errors that would otherwise be hidden by the "
            "fall-back-to-last-homography recovery logic."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to Python config file (if not specified, uses defaults)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Additionally request deterministic CUDA kernels (cuDNN/cuBLAS) before "
            "running (see flatsam2.determinism). Off by default — carries a runtime "
            "penalty (autotuner off); use it to compare outputs bit-for-bit across "
            "runs. RNG seeding happens regardless (see --seed)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for all RNGs (Python/NumPy/OpenCV/PyTorch). Applied by default — "
            "no runtime penalty — so RANSAC and other RNG-driven steps are "
            "reproducible. Also seeds the --deterministic kernel path."
        ),
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser()

    if args.deterministic:
        from flatsam2.determinism import set_global_determinism

        set_global_determinism(args.seed)
    else:
        from flatsam2.determinism import set_reproducible_seeds

        set_reproducible_seeds(args.seed)

    config = None
    if args.config:
        try:
            config = load_config_from_file(args.config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"ERROR loading config from {args.config}: {e}")
            return 1

    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR

    if not args.base_dir.exists():
        print(f"ERROR: Dataset directory not found: {args.base_dir}")
        return 1

    try:
        sequences = get_sequences(
            args.dataset,
            args.base_dir,
            sequences=args.sequences,
            subset_file=args.subset_file,
            split=args.split,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    failed = run_experiments(
        args.dataset,
        args.base_dir,
        args.output_dir,
        sequences,
        args.split,
        args.save_masks,
        not args.no_skip_completed,
        config,
        args.config,
        fail_hard=args.fail_hard,
        initfix_dir=args.initfix_dir,
    )
    print("\n" + "=" * 60)
    print("Tracking experiments completed!")
    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} sequence(s) did not produce predictions.")
        return 1
    return 0


if __name__ == "__main__":
    # Exit status must reflect the result, not merely that the process reached
    # the end, so a chained evaluation does not run on a partial result.
    sys.exit(main())
