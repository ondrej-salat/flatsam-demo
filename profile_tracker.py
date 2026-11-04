"""Timing harness for the FlatSAM2 tracker.

Runs a tracker config over a single sequence (or the first N frames of it) with
the section profiler enabled, then prints a per-building-block timing report:
SAM3, LoMa keypoints (CPU prep vs GPU detect/describe), LoMa match, RANSAC,
Hough mask-H, symmetry, and optical flow. GPU time is attributed correctly via
``torch.cuda.synchronize`` at section boundaries.

Results go to a throwaway temporary directory by default so real result trees
are never polluted.

Examples::

    python profile_tracker.py pot --config configs/tracker/loma_samh.py
    python profile_tracker.py pot --config configs/tracker/loma_samh.py \
        --sequence V01_1 --max-frames 60 --warmup 5
    python profile_tracker.py pot --config configs/tracker/loma_samh.py --no-sync
"""

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from flatsam2.config import compute_output_dir, load_config_from_file
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    make_gt_loader,
    make_image_paths,
    warp_corners,
)
from flatsam2.path_config import paths
from flatsam2.profiling import profiler
from flatsam2.tracker import FlatSAMTracker, TrackResult
from run import save_corners


def _model_device(model: Optional[object]) -> str:
    """Return the device string of a torch module, or ``'-'`` when unavailable."""
    if model is None:
        return "-"
    try:
        return str(next(iter(model.parameters())).device)
    except (StopIteration, AttributeError):
        return "?"


def _report_devices(tracker: FlatSAMTracker) -> str:
    """Summarize which sub-models sit on which device.

    Args:
        tracker: An initialised tracker.

    Returns:
        One-line device summary.
    """
    loma_model = tracker._loma.model if tracker._loma is not None else None
    parts = [
        f"cuda_available={torch.cuda.is_available()}",
        f"flow={_model_device(tracker.flower)}",
        f"loma={_model_device(loma_model)}",
        f"dino={_model_device(tracker._dino.model if tracker._dino else None)}",
    ]
    return "Devices: " + ", ".join(parts)


def profile_sequence(
    tracker: FlatSAMTracker,
    image_paths: list,
    init_corners,
    output_dir: Path,
    seq_name: str,
    max_frames: Optional[int],
    warmup: int,
) -> int:
    """Run one sequence under the profiler and save corners to *output_dir*.

    Stats are cleared after the warm-up frames so one-off costs (cuDNN autotune,
    lazy model loads, allocator growth) do not skew steady-state numbers.

    Args:
        tracker: Tracker to drive.
        image_paths: Ordered frame paths for the sequence.
        init_corners: Initial corner coordinates (2, 4).
        output_dir: Directory to write the predicted corners into.
        seq_name: Sequence name (for the output filename).
        max_frames: Stop after this many frames total (including init); ``None``
            runs the whole sequence.
        warmup: Number of tracked frames to time-then-discard before measuring.

    Returns:
        Number of frames actually included in the timing stats.
    """
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    n_timed = 0
    all_corners = []
    for frame_idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Failed to load image: {img_path}")

        if frame_idx == 0:
            tracker.init(frame, init_corners=init_corners)
            all_corners.append(init_corners)
            continue

        # Drop warm-up measurements right before the first timed frame.
        if frame_idx == warmup + 1:
            profiler.reset()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        with profiler.section("frame"):
            result: TrackResult = tracker.track(frame)

        all_corners.append(warp_corners(result.homography, init_corners))
        if frame_idx > warmup:
            n_timed += 1

    save_corners(np.stack(all_corners), output_dir / f"{seq_name}.txt")
    return n_timed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset key")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Python tracker config file",
    )
    parser.add_argument(
        "--sequence",
        help="Sequence to profile (defaults to the first sequence of the dataset)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Profile only the first N frames of the sequence (0 = whole sequence)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Tracked frames to discard before measuring (cuDNN autotune, lazy loads)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Throwaway output dir (default: a fresh temp dir under the system tmp)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Do not torch.cuda.synchronize at section boundaries (measures CPU-launch time only)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Dataset base directory (uses path_config.py default if not specified)",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help="initfix directory for the 'pti' GT loader (defaults to path_config.PT_INITFIX_DIR)",
    )
    args = parser.parse_args()

    config = load_config_from_file(args.config)
    print(f"Loaded config from: {args.config}")

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    initfix_dir = args.initfix_dir
    if ds_info.gt_loader == "pti" and initfix_dir is None:
        initfix_dir = paths.PT_INITFIX_DIR
    if not base_dir.exists():
        print(f"ERROR: Dataset directory not found: {base_dir}")
        return 1

    sequences = get_sequences(args.dataset, base_dir)
    seq_name = args.sequence or sequences[0]
    if seq_name not in sequences:
        print(f"ERROR: sequence {seq_name!r} not found in {ds_info.label}")
        return 1

    load_gt_fn = make_gt_loader(args.dataset, initfix_dir)
    img_paths_fn = make_image_paths(args.dataset)
    gt_corners, _valid = load_gt_fn(base_dir, seq_name)
    image_paths = img_paths_fn(base_dir, seq_name)
    init_corners = gt_corners[0]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="flatsam2_profile_"))
    else:
        output_dir = compute_output_dir(output_dir, args.dataset, args.config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the tracker (model loads excluded from timing by the warm-up reset).
    tracker = FlatSAMTracker(config, fail_hard=True)

    max_frames = args.max_frames if args.max_frames > 0 else None
    profiler.enable(sync_cuda=not args.no_sync)
    print(f"Profiling {ds_info.label}/{seq_name} -> {output_dir}")
    print(_report_devices(tracker))

    n_timed = profile_sequence(
        tracker,
        image_paths,
        init_corners,
        output_dir,
        seq_name,
        max_frames,
        args.warmup,
    )

    sync_state = "off" if args.no_sync else "on"
    print(
        f"\n=== Timing report ({args.config.stem} on {ds_info.label}/{seq_name}, "
        f"{n_timed} frames, cuda sync={sync_state}) ==="
    )
    print(profiler.report(n_frames=n_timed))
    print(f"\nThrowaway results in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
