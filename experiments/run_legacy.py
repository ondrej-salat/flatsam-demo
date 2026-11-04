"""Run the legacy flatsam trackers (WOFTSAM, SAM-H) on flatsam2's datasets.

The WACV/ECCV-era SAM2-based trackers live in a separate repository
(``~/git/flatsam``, "flatsam1").  They are frozen — we do not port them.  This
script keeps flatsam2 as the source of truth for sequence discovery, ground
truth, init corners and output layout, and uses the legacy package purely as the
tracker: legacy config -> SAM2 predictor -> ``track_function`` generator ->
per-frame homography -> warped corners in flatsam2's prediction format.

That split matters because flatsam2's loaders fix defects the legacy ones never
had to face (TMT frame/annotation off-by-one, per-instance MPOT-3K tasks, UCSB
which legacy does not know at all), so the baselines are produced on exactly the
same data the flatsam2 tracker sees.

Memory: the legacy tracker takes a whole sequence as an in-memory list of decoded
frames, so a POT-210 sequence alone is ~3 GB of RAM on top of SAM2's state.  Submit
with ``--mem 48G`` per work part; 16 G is OOM-killed.

Environment: this must run under the **legacy** venv (``~/git/flatsam/venv``),
which has ``sam2`` and ``kornia``; the flatsam2 venv does not.  It satisfies
flatsam2's own dependencies too, so importing both sides at once works.  Use
``slurm/legacy_env.sh`` (or ``slurm/submit.py --legacy-config``) to get it.

Results land in the ordinary flatsam2 layout, ``<output-dir>/<name>/<label>/``,
with ``<name>`` defaulting to the legacy config's stem — the convention the
already-imported legacy runs (``woftsam_samfallback_nl/``, ``SAM_H/``) and
``configs/aliases/default.py`` already use, so ``evaluate.py`` and
``collect_results.py`` need no special casing.

Examples:
    python experiments/run_legacy.py pot \\
        --legacy-config configs/woftsam_samfallback_nl.py
    python experiments/run_legacy.py tmt \\
        --legacy-config configs/WACV2026-SAM-H.py --name SAM-H_tmt
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from einops import rearrange

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.datasets import (  # noqa: E402
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    make_gt_loader,
    make_image_paths,
    warp_corners,
)
from flatsam2.path_config import paths  # noqa: E402

DEFAULT_FLATSAM1_ROOT = Path("~/git/flatsam")


def load_legacy_config(flatsam1_root: Path, config_path: Path) -> Any:
    """Load a legacy flatsam config, with the legacy repo as the working dir.

    Legacy configs compose via relative paths (``load_config('configs/sam/...')``)
    and the SAM2 checkpoints are addressed as ``./checkpoints/...``, so the
    process CWD must be the legacy repo root from here on.

    Args:
        flatsam1_root: Legacy repository root (holds ``configs/``, ``checkpoints/``).
        config_path: Legacy config path, absolute or relative to *flatsam1_root*.

    Returns:
        The legacy ``Config`` object.
    """
    os.chdir(flatsam1_root)
    sys.path.insert(0, str(flatsam1_root / "src"))
    from flatsam.config import load_config

    return load_config(config_path)


def build_legacy_predictor(conf: Any) -> Any:
    """Build the segmentation predictor a legacy config asks for.

    Mirrors ``flatsam/experiments/pt_sam_run.py``: SAM2.1 video predictor unless
    the config selects SAM3.

    Args:
        conf: Legacy config with a ``sam`` sub-config.

    Returns:
        The predictor object consumed by the legacy track functions.
    """
    from flatsam.flatsam import get_predictor, get_sam3_predictor

    if conf.sam.model == "sam3":
        return get_sam3_predictor()
    return get_predictor(
        conf.sam.size, conf.sam.memory_stride, conf.sam.do_not_update_when_not_present
    )


def guard_sam3_autocast() -> None:
    """Stop SAM3's global bf16 autocast from leaking into the legacy tracker.

    ``sam3.model.sam3_tracking_predictor`` enters ``torch.autocast(bfloat16)`` in its
    constructor and never exits it ("keep using for the entire model process"), so once a
    SAM3 predictor exists, every later CUDA op in the process — the legacy tracker's optical
    flow and its least-squares homography included — runs in bfloat16.  The SAM2 path is
    immune because ``flatsam.flatsam.sam_track`` yields inside
    ``torch.amp.autocast('cuda', enabled=False)``; the SAM3 adapter has no such guard.

    Wraps ``flatsam.flatsam.sam3_track`` so the consumer side of the generator runs with
    autocast disabled and SAM3 itself keeps its bf16 context, mirroring ``sam_track``.
    """
    import torch
    from flatsam import flatsam as legacy_flatsam

    original = legacy_flatsam.sam3_track

    def guarded(*args: Any, **kwargs: Any):
        generator = original(*args, **kwargs)
        while True:
            try:
                item = next(generator)
            except StopIteration:
                return
            with torch.amp.autocast("cuda", enabled=False):
                yield item

    legacy_flatsam.sam3_track = guarded


def build_precomputed_predictor(masks_root: Path) -> Any:
    """Build a fake SAM predictor that replays masks dumped by an earlier run.

    Mimics the SAM2 video-predictor interface consumed by ``flatsam.flatsam.sam_track``
    (``init_state`` / ``add_new_mask`` / ``propagate_in_video``), so a run driven by
    it takes the *same* code path as a live SAM2 run and differs only in mask content.
    That isolates "which masks" from "which segmentation code path".

    Masks are read from ``masks_root/<seq_name>/<NNNNN>.png`` with the 1-based frame
    numbering the legacy runners (``pot_sam_run.py``, ``pt_sam_run.py``) write.

    Args:
        masks_root: Directory holding one subdirectory of PNG masks per sequence.

    Returns:
        A predictor object accepted by the legacy ``sam_track``.

    Raises:
        ValueError: If a mask is missing for a frame the tracker asks for.
    """
    import torch

    class PrecomputedSAM:
        def __init__(self, masks_root: Path) -> None:
            self.masks_root = Path(masks_root)
            self.video_path: Optional[Path] = None
            self.seq_name: Optional[str] = None

        def init_state(self, video_path, offload_video_to_cpu=None, async_loading_frames=None):
            self.video_path = Path(video_path)
            self.seq_name = self.video_path.stem
            return None

        def add_new_mask(self, *args, **kwargs):
            return None, None, None

        def propagate_in_video(self, state):
            frame_paths = sorted(self.video_path.glob("*.jpg"))
            for frame_idx, frame_path in enumerate(frame_paths):
                mask_path = self.masks_root / self.seq_name / f"{frame_path.stem}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError(f"missing precomputed mask: {mask_path}")
                logits = torch.from_numpy(
                    rearrange(mask.astype(np.float32) - 127.5, "H W -> 1 H W")
                )
                yield frame_idx, [1], [logits]

    return PrecomputedSAM(masks_root)


def track_sequence(
    predictor: Any,
    conf: Any,
    image_paths: List[Path],
    init_corners: np.ndarray,
    seq_name: str,
) -> np.ndarray:
    """Track one sequence with the legacy tracker.

    Args:
        predictor: Legacy SAM predictor.
        conf: Legacy config; ``conf.track_function`` selects the tracker
            (absent -> plain ``flatsam_track``, i.e. the SAM-H family).
        image_paths: Ordered frame paths.
        init_corners: (2, 4) init-frame corners.
        seq_name: Sequence name (legacy code uses it for logging only).

    Returns:
        (F, 2, 4) predicted corners, one entry per frame.
    """
    from flatsam.flatsam import flatsam_track

    frames = [cv2.imread(str(path)) for path in image_paths]

    track_function = conf.track_function if conf.track_function else flatsam_track

    all_corners: List[np.ndarray] = []
    for _, _, _, debug_info in track_function(
        predictor, conf, frames, init_corners, seq_name, False, None
    ):
        try:
            if "output_H" in debug_info:
                H_init2current = debug_info["output_H"]
            else:
                H_init2current = np.linalg.inv(debug_info["output_H2init"])
            all_corners.append(warp_corners(H_init2current, init_corners))
        except Exception:
            # Singular H — reuse the last good pose, exactly like the legacy runner.
            all_corners.append(all_corners[-1] if all_corners else init_corners.copy())

    return np.stack(all_corners, axis=0)


def save_corners(corners: np.ndarray, output_path: Path) -> None:
    """Write predictions in flatsam2's 8-floats-per-line format.

    Args:
        corners: (F, 2, 4) predicted corners.
        output_path: Destination ``<seq_name>.txt``.
    """
    corners_flat = rearrange(
        corners, "frames xy corners -> frames (corners xy)", corners=4, xy=2
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, corners_flat, fmt="%.6f")


def is_sequence_complete(output_dir: Path, seq_name: str, num_frames: int) -> bool:
    """Whether a sequence already has a full-length prediction file.

    Args:
        output_dir: Per-config/dataset output directory.
        seq_name: Sequence name.
        num_frames: Expected number of frames.

    Returns:
        True if ``<seq_name>.txt`` exists with one row per frame.
    """
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


def save_legacy_run_info(
    output_dir: Path, flatsam1_root: Path, config_path: Path, dataset: str,
    precomputed_masks: Optional[Path] = None,
) -> None:
    """Record which legacy code produced these results.

    flatsam2's ``reproducibility.save_run_info`` expects a ``TrackerConfig``, so
    legacy runs get their own minimal sidecar: the legacy config, its resolved
    text, and the legacy repo's git commit.

    Args:
        output_dir: Per-config/dataset output directory.
        flatsam1_root: Legacy repository root.
        config_path: Legacy config path (absolute).
        dataset: Dataset CLI key.
        precomputed_masks: Mask directory replayed instead of live segmentation, if any.
    """
    repro_dir = output_dir / ".repro"
    repro_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(
            ["git", "-C", str(flatsam1_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except Exception:
        commit = ""

    info = {
        "runner": "experiments/run_legacy.py",
        "dataset": dataset,
        "flatsam1_root": str(flatsam1_root),
        "legacy_config": str(config_path),
        "legacy_git_commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "precomputed_masks": str(precomputed_masks) if precomputed_masks else None,
    }
    (repro_dir / "legacy_run.json").write_text(json.dumps(info, indent=2))
    try:
        (repro_dir / "legacy_config.py").write_text(config_path.read_text())
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to run on.")
    parser.add_argument(
        "--legacy-config", type=Path, required=True,
        help="Legacy flatsam config (absolute, or relative to --flatsam1-root).",
    )
    parser.add_argument(
        "--flatsam1-root", type=Path, default=DEFAULT_FLATSAM1_ROOT,
        help="Legacy flatsam repository root.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Output directory stem (default: the legacy config's stem).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=paths.OUTPUT_DIRECTORY,
        help="Top-level results directory.",
    )
    parser.add_argument("--base-dir", type=Path, default=None, help="Dataset root override.")
    parser.add_argument("--split", choices=["train", "test"], default=None, help="Dataset split.")
    parser.add_argument("--sequences", nargs="+", default=None, help="Explicit sequence names.")
    parser.add_argument("--subset", default=None, help="Subset name in configs/subsets/.")
    parser.add_argument("--subset-file", type=Path, default=None, help="Path to a subset file.")
    parser.add_argument(
        "--initfix-dir", type=Path, default=None,
        help="Reannotated init corners (PTi; defaults to path_config).",
    )
    parser.add_argument(
        "--guard-autocast", action="store_true",
        help="Wrap the SAM3 mask generator so the tracker's own CUDA work runs outside "
             "SAM3's never-exited global bf16 autocast (what sam_track does for SAM2).",
    )
    parser.add_argument(
        "--disable-tf32", action="store_true",
        help="Turn TF32 matmul/cuDNN back off after the predictor is built. SAM3's "
             "model_builder enables TF32 process-wide on import, which silently "
             "degrades the flow and homography solves of the legacy tracker.",
    )
    parser.add_argument(
        "--precomputed-masks", type=Path, default=None,
        help="Replay masks dumped by an earlier run instead of segmenting "
             "(<dir>/<seq_name>/<NNNNN>.png, 1-based); forces the SAM2 code path.",
    )
    parser.add_argument(
        "--no-skip-completed", action="store_true",
        help="Recompute sequences that already have a complete prediction file.",
    )
    parser.add_argument(
        "--fail-hard", action="store_true",
        help="Re-raise per-sequence errors instead of continuing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    flatsam1_root = args.flatsam1_root.expanduser().resolve()
    if not (flatsam1_root / "src" / "flatsam").is_dir():
        print(f"ERROR: no legacy flatsam package under {flatsam1_root}")
        return 1

    config_path = args.legacy_config
    if not config_path.is_absolute():
        config_path = flatsam1_root / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        print(f"ERROR: legacy config not found: {config_path}")
        return 1

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    base_dir = Path(base_dir).expanduser().resolve()
    initfix_dir = args.initfix_dir
    if ds_info.gt_loader == "pti" and initfix_dir is None:
        initfix_dir = paths.PT_INITFIX_DIR
    if not base_dir.exists():
        print(f"ERROR: dataset directory not found: {base_dir}")
        return 1

    # Resolve everything that is relative to the flatsam2 repo *before* the
    # chdir into the legacy repo that config loading requires.
    try:
        sequences = get_sequences(
            args.dataset, base_dir,
            sequences=args.sequences, subset=args.subset,
            subset_file=args.subset_file, split=args.split,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    name = args.name or config_path.stem
    output_dir = Path(args.output_dir).expanduser().resolve() / name / ds_info.label
    output_dir.mkdir(parents=True, exist_ok=True)

    load_gt_fn = make_gt_loader(args.dataset, initfix_dir)
    img_paths_fn = make_image_paths(args.dataset)

    masks_root = args.precomputed_masks
    if masks_root is not None:
        masks_root = Path(masks_root).expanduser().resolve()
        if not masks_root.is_dir():
            print(f"ERROR: precomputed mask directory not found: {masks_root}")
            return 1

    save_legacy_run_info(output_dir, flatsam1_root, config_path, args.dataset, masks_root)

    conf = load_legacy_config(flatsam1_root, config_path)
    if masks_root is not None:
        predictor = build_precomputed_predictor(masks_root)
    else:
        predictor = build_legacy_predictor(conf)

    if args.guard_autocast:
        guard_sam3_autocast()
        print("Autocast guard  : SAM3's bf16 context no longer covers the tracker")

    if args.disable_tf32:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("TF32 disabled   : matmul and cuDNN back to full float32")

    print(f"Legacy config   : {config_path}")
    if masks_root is not None:
        print(f"Precomputed masks: {masks_root}")
    print(f"Dataset         : {ds_info.label} ({len(sequences)} sequences)")
    print(f"Output directory: {output_dir}")

    start_time = time.time()
    for seq_i, seq_name in enumerate(sequences, start=1):
        print(f"\n{'=' * 60}\nSequence: {seq_name}  [{seq_i}/{len(sequences)}]\n{'=' * 60}")
        image_paths = img_paths_fn(base_dir, seq_name)
        if not args.no_skip_completed and is_sequence_complete(
            output_dir, seq_name, len(image_paths)
        ):
            print(f"Skipping {seq_name} (already complete)")
            continue

        seq_start = time.time()
        try:
            gt_corners, _ = load_gt_fn(base_dir, seq_name)
            pred_corners = track_sequence(
                predictor, conf, image_paths, gt_corners[0], seq_name
            )
            save_corners(pred_corners, output_dir / f"{seq_name}.txt")
            print(f"Saved results to: {output_dir / f'{seq_name}.txt'}")
        except Exception as e:
            if args.fail_hard:
                raise
            print(f"ERROR processing {seq_name}: {e}")
            import traceback

            traceback.print_exc()
        seq_elapsed = time.time() - seq_start
        fps = len(image_paths) / seq_elapsed if seq_elapsed > 0 else 0.0
        print(f"Sequence time: {seq_elapsed:.1f}s ({fps:.1f} FPS)")

    print(f"\nTotal elapsed: {time.time() - start_time:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
