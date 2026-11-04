"""Collect real-world validation data for NN homography quality estimation.

Extracts mask/homography/GT-RMSE triples from PlanarTrack tracking results,
storing the raw ingredients so that any architecture's features can be
computed later.

Usage examples:

    # Collect from one config, PlanarTrack train split:
    python -m flatsam2.nn.collect_realworld_dataset \\
        --config-names default \\
        --output-dir /scratch/datasets/realworld_val

    # Collect from multiple configs, reuse masks from a different config:
    python -m flatsam2.nn.collect_realworld_dataset \\
        --config-names fast_kitti flow_hough_symmetry_nn_cascade \\
        --masks-config fast_kitti \\
        --output-dir /scratch/datasets/realworld_val

    # Custom subset sizes:
    python -m flatsam2.nn.collect_realworld_dataset \\
        --config-names default \\
        --subset-sizes 100,500,2000 \\
        --output-dir /scratch/datasets/realworld_val

Output layout::

    output_dir/
      meta.json               # generation metadata
      sequences.json          # per-sequence info (name, init_corners, image_shape)
      masks/                  # copied mask PNGs
        {seq_name}/
          00001.png           # init mask
          {frame+1:05d}.png   # current-frame masks
      samples.npz             # numerical data (seq_ids, frame_indices,
                              #   homographies, gt_rmse, sources, config_names)
      subsets/
        subset_200.json       # stratified subset indices + bin stats
        subset_1000.json

The stored sample pool is bounded to the largest requested subset size by
collecting samples online into RMSE bins after a global frame shuffle.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from einops import rearrange
from numpy.lib.npyio import NpzFile

from ..config import compute_output_dir
from ..datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_load_gt,
    pt_load_gt,
    pti_load_gt,
    warp_corners,
)
from ..homography_quality import homography_from_corners
from ..metrics import alignment_error
from ..path_config import paths


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANDIDATE_NAMES = ("flow_H", "mask_H", "redetection_flow_H")

_DEFAULT_BIN_EDGES = np.array(
    # Bins are aligned with the decision thresholds used in quality_threshold_analysis.py
    # (t_good ≈ 12px, t_bad ≈ 25px) and respect the GT noise floor of ~5.71px
    # (PlanarTrack authors report mean GT annotation error ≈ 5.71px, 10.71% > 15px).
    # Fine bins below 5px are omitted — they are below the GT noise floor and
    # will be nearly empty in practice.
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, np.inf]
)


# ---------------------------------------------------------------------------
# GT loading helper (same pattern as evaluate.py / nn_quality_eval.py)
# ---------------------------------------------------------------------------


def _make_load_gt_fn(dataset: str, initfix_dir: Optional[Path]):
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        raise ValueError("initfix_dir must be provided for the 'pti' GT loader")
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)


def _load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    """Load predicted corners (N, 2, 4) from a results directory."""
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    return rearrange(
        pred_flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2
    )


def _load_debug_data(
    results_dir: Path, seq_name: str
) -> Tuple[Optional[NpzFile], Dict[int, int]]:
    """Load debug candidate homographies and build a frame index map."""
    debug_path = results_dir / f"{seq_name}_debug.npz"
    if not debug_path.exists():
        return None, {}

    try:
        debug_data = np.load(debug_path, allow_pickle=False)
    except Exception:
        return None, {}

    debug_frame_map: Dict[int, int] = {}
    if "frame_idx" in debug_data:
        for row_i, frame_idx in enumerate(debug_data["frame_idx"]):
            debug_frame_map[int(frame_idx)] = row_i

    return debug_data, debug_frame_map


# ---------------------------------------------------------------------------
# Per-frame sample collection
# ---------------------------------------------------------------------------


def collect_frame_samples(
    seq_name: str,
    config_name: str,
    gt_corners: np.ndarray,
    pred_corners: np.ndarray,
    frame_idx: int,
    debug_data: Optional[NpzFile],
    debug_frame_map: Dict[int, int],
) -> List[dict]:
    """Collect all sample candidates for a single frame.

    Collects the "predicted" homography (recovered from corners) for the
    requested frame, plus candidate homographies from debug.npz when available.

    Args:
        seq_name: Sequence identifier.
        config_name: Tracker config name (for provenance).
        gt_corners: (N, 2, 4) ground-truth corner coordinates.
        pred_corners: (N, 2, 4) predicted corner coordinates.
        frame_idx: Frame index to collect.
        debug_data: Loaded ``debug.npz`` object, if available.
        debug_frame_map: Mapping from frame index to debug row.

    Returns:
        List of sample dicts, each with keys: seq_name, config_name,
        frame_idx, H, gt_rmse, source.
    """
    init_corners = gt_corners[0]  # (2, 4)
    samples: List[dict] = []

    has_debug_for_frame = debug_data is not None and frame_idx in debug_frame_map
    if not has_debug_for_frame:
        H_pred = homography_from_corners(init_corners, pred_corners[frame_idx])
        if H_pred is not None:
            gt_rmse = float(alignment_error(gt_corners[frame_idx], pred_corners[frame_idx]))
            samples.append(
                {
                    "seq_name": seq_name,
                    "config_name": config_name,
                    "frame_idx": frame_idx,
                    "H": H_pred,
                    "gt_rmse": gt_rmse,
                    "source": "predicted",
                }
            )

    if has_debug_for_frame:
        row = debug_frame_map[frame_idx]
        for cand_name in _CANDIDATE_NAMES:
            if cand_name not in debug_data:
                continue
            H_cand = debug_data[cand_name][row]  # (3, 3)
            if np.any(np.isnan(H_cand)):
                continue

            warped = warp_corners(H_cand, init_corners)
            gt_rmse_cand = float(alignment_error(gt_corners[frame_idx], warped))
            samples.append(
                {
                    "seq_name": seq_name,
                    "config_name": config_name,
                    "frame_idx": frame_idx,
                    "H": H_cand,
                    "gt_rmse": gt_rmse_cand,
                    "source": cand_name,
                }
            )

    return samples


def _format_bin_label(lo: float, hi: float) -> str:
    """Format an RMSE bin label."""
    return f"[{lo:.0f}, {hi:.0f})" if np.isfinite(hi) else f"[{lo:.0f}, inf)"


def _link_or_copy(src: Path, dst: Path) -> bool:
    """Create a hardlink when possible, otherwise copy.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        True if a new file/link was created, False if destination already exists.
    """
    if dst.exists():
        return False

    try:
        os.link(src, dst)
        return True
    except FileExistsError:
        return False
    except OSError:
        shutil.copy2(src, dst)
        return True


# ---------------------------------------------------------------------------
# Stratified subset creation and sequence-disjoint splitting
# ---------------------------------------------------------------------------


def _assign_sequences_to_splits(
    seq_ids: np.ndarray,
    gt_rmse: np.ndarray,
    split_sizes: List[int],
    rng: np.random.Generator,
) -> Dict[int, int]:
    """Assign each unique sequence to exactly one split, proportional to split sizes.

    Sequences are sorted by their mean GT RMSE and then assigned using a
    deficit-based (Bresenham-style) interleaving: at each step the split most
    behind its target proportion receives the next sequence.  This simultaneously
    achieves two goals:

    1. **Proportional allocation** — each split receives a number of sequences
       proportional to its target sample size, so frame utilisation is roughly
       equal across splits (no split wastes 90% of its frames or is forced to
       use 85%).

    2. **RMSE-balanced interleaving** — because assignment interleaves across
       the RMSE-sorted sequence list, every split gets a representative mix of
       easy and hard sequences rather than contiguous RMSE blocks.

    Args:
        seq_ids: (N,) integer sequence IDs for each sample.
        gt_rmse: (N,) GT RMSE values.
        split_sizes: Target sample counts for each split.  Sequence counts are
            allocated proportionally to these sizes.
        rng: Random generator used to break ties within RMSE-rank blocks.

    Returns:
        Dict mapping sequence_id -> split_index (0-based).
    """
    unique_seqs = np.unique(seq_ids)
    n_seqs = len(unique_seqs)
    seq_mean_rmse = np.array(
        [float(np.mean(gt_rmse[seq_ids == s])) for s in unique_seqs]
    )

    # Add small random jitter to break RMSE ties reproducibly
    jitter = rng.uniform(0.0, 1e-6, size=n_seqs)
    sorted_order = np.argsort(seq_mean_rmse + jitter, kind="stable")
    sorted_seqs = unique_seqs[sorted_order]

    # Deficit-based proportional interleaving:
    # target_fraction[j] = split_sizes[j] / sum(split_sizes)
    # At step i, assign to the split j with the largest
    #   deficit = target_fraction[j] - (sequences assigned to j so far) / (i+1)
    total = float(sum(split_sizes))
    target_fractions = [s / total for s in split_sizes]
    counts = [0] * len(split_sizes)
    assignment: Dict[int, int] = {}

    for i, seq_id in enumerate(sorted_seqs):
        deficits = [
            target_fractions[j] - counts[j] / (i + 1) for j in range(len(split_sizes))
        ]
        j = int(np.argmax(deficits))
        assignment[int(seq_id)] = j
        counts[j] += 1

    return assignment


def create_stratified_subset(
    gt_rmse: np.ndarray,
    subset_size: int,
    bin_edges: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, dict]:
    """Select a well-distributed subset based on GT RMSE bins.

    Allocates equally across non-empty bins.  If a bin has fewer samples
    than its quota, takes all and redistributes the remainder to other bins.

    Args:
        gt_rmse: (N,) float32 array of GT RMSE values.
        subset_size: Target number of samples.
        bin_edges: Sorted array of bin edges (e.g. [0, 2, 5, 10, 15, 20, 30, inf]).
        rng: Numpy random generator.

    Returns:
        (selected_indices, bin_stats) where bin_stats maps bin label to
        count of selected samples.
    """
    n_total = len(gt_rmse)
    n_bins = len(bin_edges) - 1
    bin_indices = np.digitize(gt_rmse, bin_edges) - 1  # 0-based bin id

    if subset_size >= n_total:
        bin_stats: dict = {}
        for b in range(n_bins):
            count_total = int(np.sum(bin_indices == b))
            if count_total == 0:
                continue
            label = _format_bin_label(bin_edges[b], bin_edges[b + 1])
            bin_stats[label] = {
                "total": count_total,
                "selected": count_total,
            }
        return np.arange(n_total), bin_stats

    # Group sample indices by bin
    bins: Dict[int, np.ndarray] = {}
    for b in range(n_bins):
        members = np.where(bin_indices == b)[0]
        if len(members) > 0:
            bins[b] = members

    n_nonempty = len(bins)
    if n_nonempty == 0:
        return np.array([], dtype=np.int64), {}

    # Iteratively allocate: equal quota, take all from small bins, redistribute
    remaining_target = subset_size
    selected: List[np.ndarray] = []
    available_bins = dict(bins)  # bin_id -> member indices

    while remaining_target > 0 and available_bins:
        quota = max(1, remaining_target // len(available_bins))
        exhausted = []
        allocated_this_round = 0

        for b, members in available_bins.items():
            take = min(quota, len(members), remaining_target - allocated_this_round)
            if take <= 0:
                continue
            chosen = rng.choice(members, size=take, replace=False)
            selected.append(chosen)
            allocated_this_round += take

            leftover = np.setdiff1d(members, chosen)
            if len(leftover) == 0:
                exhausted.append(b)
            else:
                available_bins[b] = leftover

        remaining_target -= allocated_this_round
        for b in exhausted:
            del available_bins[b]

    all_selected = (
        np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    )
    all_selected = np.unique(all_selected)  # deduplicate, sorted

    # Build bin stats
    bin_stats: dict = {}
    selected_bins = np.digitize(gt_rmse[all_selected], bin_edges) - 1
    for b in range(n_bins):
        label = _format_bin_label(bin_edges[b], bin_edges[b + 1])
        count_total = int(np.sum(bin_indices == b))
        count_selected = int(np.sum(selected_bins == b))
        if count_total > 0:
            bin_stats[label] = {
                "total": count_total,
                "selected": count_selected,
            }

    return all_selected, bin_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect real-world validation data for NN quality estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Base results directory (default: paths.OUTPUT_DIRECTORY).",
    )
    parser.add_argument(
        "--config-names",
        nargs="+",
        required=True,
        help="Config subdirectory name(s) to collect from.",
    )
    parser.add_argument(
        "--dataset",
        default="planartrack",
        choices=CLI_CHOICES,
        help="Dataset identifier.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="PlanarTrack split.",
    )
    parser.add_argument(
        "--masks-from",
        type=Path,
        default=None,
        help="Separate results dir for masks (if omitted, uses --results-dir).",
    )
    parser.add_argument(
        "--masks-config",
        default=None,
        help="Config name whose masks/ dir to use (if different from --config-names).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the collected dataset.",
    )
    parser.add_argument(
        "--subset-sizes",
        default="10000,30000,100000",
        help=(
            "Comma-separated split sizes.  Splits are sequence-disjoint: "
            "all frames from a sequence land in exactly one split.  "
            "Intended use: split_10000=calibration, split_30000=validation, "
            "split_100000=held-out test."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for subset sampling.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Dataset base directory (default: from path_config).",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help="Initfix directory (default: from path_config for pti datasets).",
    )
    args = parser.parse_args()

    # --- Resolve paths ---
    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY
    masks_base = args.masks_from if args.masks_from is not None else args.results_dir

    load_gt_fn = _make_load_gt_fn(args.dataset, args.initfix_dir)
    subset_sizes = sorted(
        {int(s.strip()) for s in args.subset_sizes.split(",") if s.strip()}
    )
    if not subset_sizes or subset_sizes[0] <= 0:
        raise ValueError("--subset-sizes must contain at least one positive integer")

    collection_rng = np.random.default_rng(args.seed)

    # --- Discover sequences ---
    sequences = get_sequences(
        args.dataset,
        args.base_dir,
        split=args.split,
    )
    print(
        f"Dataset: {ds_info.label}  split={args.split}  "
        f"sequences={len(sequences)}  configs={args.config_names}"
    )
    print(
        f"Target splits: {subset_sizes} (total {sum(subset_sizes):,}); collecting all available frames"
    )

    # --- Phase 1: Index available frame results ---
    t0 = time.perf_counter()
    candidate_frames: List[Tuple[str, str, int]] = []
    sequence_payloads: Dict[Tuple[str, str], dict] = {}
    seq_info_map: Dict[str, dict] = {}  # seq_name -> {init_corners, image_shape}
    skipped_no_pred = 0
    skipped_no_masks = 0

    for config_name in args.config_names:
        results_dir = compute_output_dir(
            args.results_dir,
            args.dataset,
            Path(f"{config_name}.py"),
        )
        masks_config = args.masks_config if args.masks_config else config_name
        masks_dir = compute_output_dir(
            masks_base,
            args.dataset,
            Path(f"{masks_config}.py"),
        )

        print(f"\nConfig: {config_name}")
        print(f"  Predictions: {results_dir}")
        print(f"  Masks: {masks_dir}")

        config_count = 0
        index_pbar = tqdm.tqdm(
            sequences,
            desc=f"Indexing {config_name}",
            unit="seq",
        )
        for seq_name in index_pbar:
            # Check prediction file
            pred_file = results_dir / f"{seq_name}.txt"
            if not pred_file.exists():
                skipped_no_pred += 1
                continue

            # Check init mask
            seq_masks_dir = masks_dir / "masks" / seq_name
            init_mask_path = seq_masks_dir / "00001.png"
            if not init_mask_path.exists():
                skipped_no_masks += 1
                continue

            try:
                gt_corners, valid = load_gt_fn(args.base_dir, seq_name)
                pred_corners = _load_predictions(results_dir, seq_name)
            except Exception as e:
                print(f"  {seq_name}: FAILED loading data — {e}")
                continue

            # Collect image_shape from init mask (read only once per seq)
            if seq_name not in seq_info_map:
                init_mask = cv2.imread(str(init_mask_path), cv2.IMREAD_GRAYSCALE)
                if init_mask is None:
                    print(f"  {seq_name}: FAILED reading init mask")
                    continue
                seq_info_map[seq_name] = {
                    "init_corners": gt_corners[0].tolist(),
                    "image_shape": list(init_mask.shape[:2]),
                }

            n_frames = min(len(gt_corners), len(pred_corners), len(valid))
            available_frames: List[int] = []
            for frame_idx in range(1, n_frames):
                if not valid[frame_idx]:
                    continue

                mask_path = seq_masks_dir / f"{frame_idx + 1:05d}.png"
                if not mask_path.exists():
                    continue

                available_frames.append(frame_idx)

            if not available_frames:
                continue

            sequence_payloads[(config_name, seq_name)] = {
                "results_dir": results_dir,
                "gt_corners": gt_corners,
                "pred_corners": pred_corners,
            }
            candidate_frames.extend(
                (config_name, seq_name, frame_idx) for frame_idx in available_frames
            )
            config_count += len(available_frames)

            if config_count % 20000 == 0 and config_count > 0:
                index_pbar.set_postfix({"frames": f"{config_count:,}"})

        print(f"  Indexed {config_count:,} frame results from {config_name}")

    elapsed = time.perf_counter() - t0
    print(f"\nPhase 1 done: {len(candidate_frames):,} frame results in {elapsed:.1f}s")
    if skipped_no_pred:
        print(f"  Skipped {skipped_no_pred} sequences (no prediction file)")
    if skipped_no_masks:
        print(f"  Skipped {skipped_no_masks} sequences (no masks)")

    if not candidate_frames:
        print("ERROR: No frame results found. Check paths and config names.")
        return

    # --- Phase 2: Collect all available samples ---
    # No stratified cap here — collect every valid frame result.
    # All stratification happens in Phase 5 within each sequence-disjoint split,
    # so that each split's assigned sequences have their full frame pool available.
    print("\nCollecting all available samples...")
    all_samples: List[dict] = []
    debug_loaded_count = 0
    debug_load_time_s = 0.0
    n_sequence_payloads = len(sequence_payloads)

    frame_order = collection_rng.permutation(len(candidate_frames))
    collect_pbar = tqdm.tqdm(
        frame_order,
        total=len(candidate_frames),
        desc="Collecting samples",
        unit="frame",
    )
    for processed_frames, frame_pos in enumerate(collect_pbar, start=1):
        config_name, seq_name, frame_idx = candidate_frames[int(frame_pos)]
        payload = sequence_payloads[(config_name, seq_name)]
        if "debug_loaded" not in payload:
            debug_t0 = time.perf_counter()
            debug_data, debug_frame_map = _load_debug_data(
                payload["results_dir"], seq_name
            )
            payload["debug_data"] = debug_data
            payload["debug_frame_map"] = debug_frame_map
            payload["debug_loaded"] = True
            debug_loaded_count += 1
            debug_load_time_s += time.perf_counter() - debug_t0

        frame_samples = collect_frame_samples(
            seq_name,
            config_name,
            payload["gt_corners"],
            payload["pred_corners"],
            frame_idx,
            payload["debug_data"],
            payload["debug_frame_map"],
        )
        all_samples.extend(frame_samples)

        if processed_frames % 2000 == 0:
            avg_samples_per_frame = len(all_samples) / processed_frames
            collect_pbar.set_postfix(
                {
                    "samples": f"{len(all_samples):,}",
                    "avg/frame": f"{avg_samples_per_frame:.2f}",
                    "debug": f"{debug_loaded_count}/{n_sequence_payloads}",
                    "debug_s": f"{debug_load_time_s:.1f}",
                }
            )

    for payload in sequence_payloads.values():
        debug_data = payload.get("debug_data")
        if debug_data is not None:
            debug_data.close()

    print(f"  Collected {len(all_samples):,} samples")

    if not all_samples:
        print("ERROR: No samples collected after RMSE binning.")
        return

    # --- Phase 3: Copy masks ---
    print("\nCopying masks...")
    out_masks = args.output_dir / "masks"
    frames_by_seq: Dict[str, set] = {}
    for s in all_samples:
        frames_by_seq.setdefault(s["seq_name"], set()).add(s["frame_idx"])

    masks_config = args.masks_config if args.masks_config else args.config_names[0]
    masks_dir = compute_output_dir(
        masks_base,
        args.dataset,
        Path(f"{masks_config}.py"),
    )

    copy_ops: List[Tuple[Path, Path]] = []
    prep_pbar = tqdm.tqdm(
        frames_by_seq.items(),
        total=len(frames_by_seq),
        desc="Preparing mask links",
        unit="seq",
    )
    for prepared_seqs, (seq_name, frame_indices) in enumerate(prep_pbar, start=1):
        src_masks = masks_dir / "masks" / seq_name
        dst_masks = out_masks / seq_name
        dst_masks.mkdir(parents=True, exist_ok=True)

        # Init mask
        src_init = src_masks / "00001.png"
        dst_init = dst_masks / "00001.png"
        if src_init.exists() and not dst_init.exists():
            copy_ops.append((src_init, dst_init))

        # Frame masks
        for fi in sorted(frame_indices):
            fname = f"{fi + 1:05d}.png"
            src = src_masks / fname
            dst = dst_masks / fname
            if src.exists() and not dst.exists():
                copy_ops.append((src, dst))

        if prepared_seqs % 20 == 0:
            prep_pbar.set_postfix({"ops": f"{len(copy_ops):,}"})

    masks_copied = 0
    link_pbar = tqdm.tqdm(copy_ops, desc="Linking masks", unit="file")
    for src, dst in link_pbar:
        if _link_or_copy(src, dst):
            masks_copied += 1
        if masks_copied % 2000 == 0 and masks_copied > 0:
            link_pbar.set_postfix({"linked": f"{masks_copied:,}"})

    print(f"  Created {masks_copied:,} mask links/copies")

    # --- Phase 4: Save dataset ---
    print("\nSaving dataset...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build sequence index
    seq_names_ordered = sorted({sample["seq_name"] for sample in all_samples})
    seq_name_to_id = {name: i for i, name in enumerate(seq_names_ordered)}
    sequences_list = [
        {"name": name, **seq_info_map[name]} for name in seq_names_ordered
    ]

    # Build numerical arrays
    n = len(all_samples)
    seq_ids = np.array(
        [seq_name_to_id[s["seq_name"]] for s in all_samples], dtype=np.int32
    )
    frame_indices = np.array([s["frame_idx"] for s in all_samples], dtype=np.int32)
    homographies = np.stack([s["H"] for s in all_samples]).astype(np.float64)
    gt_rmse = np.array([s["gt_rmse"] for s in all_samples], dtype=np.float32)
    sources = np.array([s["source"] for s in all_samples])
    config_names_arr = np.array([s["config_name"] for s in all_samples])

    np.savez(
        args.output_dir / "samples.npz",
        seq_ids=seq_ids,
        frame_indices=frame_indices,
        homographies=homographies,
        gt_rmse=gt_rmse,
        sources=sources,
        config_names=config_names_arr,
    )

    with open(args.output_dir / "sequences.json", "w") as f:
        json.dump(sequences_list, f, indent=2)

    # RMSE distribution summary
    rmse_finite = gt_rmse[np.isfinite(gt_rmse)]
    source_counts: Dict[str, int] = {}
    for s in sources:
        source_counts[s] = source_counts.get(s, 0) + 1
    config_counts: Dict[str, int] = {}
    for c in config_names_arr:
        config_counts[c] = config_counts.get(c, 0) + 1

    meta = {
        "n_samples": n,
        "n_sequences": len(seq_names_ordered),
        "dataset": args.dataset,
        "split": args.split,
        "config_names": args.config_names,
        "masks_config": args.masks_config,
        "seed": args.seed,
        "collection_bin_edges": _DEFAULT_BIN_EDGES.tolist(),
        "source_counts": source_counts,
        "config_counts": config_counts,
        "rmse_stats": {
            "mean": float(np.mean(rmse_finite)) if len(rmse_finite) > 0 else None,
            "median": float(np.median(rmse_finite)) if len(rmse_finite) > 0 else None,
            "min": float(np.min(rmse_finite)) if len(rmse_finite) > 0 else None,
            "max": float(np.max(rmse_finite)) if len(rmse_finite) > 0 else None,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {n:,} samples across {len(seq_names_ordered)} sequences")
    print(f"  Sources: {source_counts}")
    print(
        f"  RMSE: mean={meta['rmse_stats']['mean']:.1f}  "
        f"median={meta['rmse_stats']['median']:.1f}"
    )

    # --- Phase 5: Create sequence-disjoint stratified splits ---
    # All frames from a given sequence land in exactly one split, eliminating
    # cross-split temporal correlation (consecutive frames are near-duplicates).
    # Sequences are assigned in round-robin order sorted by mean RMSE so that
    # each split gets a balanced mix of easy and hard sequences.
    # Within each split's assigned sequences, frames are stratified by GT RMSE
    # to hit the target sample count.
    print("\nCreating sequence-disjoint stratified splits...")
    subset_rng = np.random.default_rng(args.seed)
    subsets_dir = args.output_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    seq_to_split = _assign_sequences_to_splits(
        seq_ids, gt_rmse, subset_sizes, subset_rng
    )

    # Report sequence assignment and expected frame density
    for split_idx, size in enumerate(subset_sizes):
        n_seqs_in_split = sum(1 for v in seq_to_split.values() if v == split_idx)
        candidate_frames = int(
            np.sum([seq_to_split.get(int(sid), -1) == split_idx for sid in seq_ids])
        )
        density = size / candidate_frames if candidate_frames > 0 else float("nan")
        print(
            f"  split_{size}: {n_seqs_in_split} sequences, "
            f"{candidate_frames} candidate frames, "
            f"target density {density:.1%}"
        )

    split_pbar = tqdm.tqdm(
        list(enumerate(subset_sizes)),
        desc="Building splits",
        unit="split",
    )
    for split_idx, size in split_pbar:
        # Candidate samples: all frames from sequences assigned to this split
        candidate_mask = np.array(
            [seq_to_split.get(int(sid), -1) == split_idx for sid in seq_ids]
        )
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            print(f"  split_{size}: no candidate samples, skipping")
            continue

        local_indices, bin_stats = create_stratified_subset(
            gt_rmse[candidate_indices],
            size,
            _DEFAULT_BIN_EDGES,
            subset_rng,
        )
        global_indices = candidate_indices[local_indices]

        split_data = {
            "split_size": int(len(global_indices)),
            "requested_size": size,
            "indices": global_indices.tolist(),
            "bin_stats": bin_stats,
            "sequence_disjoint": True,
            "n_sequences": int(len({int(seq_ids[i]) for i in global_indices})),
        }
        split_path = subsets_dir / f"split_{size}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)

        n_seqs_sampled = split_data["n_sequences"]
        print(
            f"  split_{size}: {len(global_indices)} samples from {n_seqs_sampled} sequences"
        )
        for label, stats in bin_stats.items():
            print(f"    {label}: {stats['selected']}/{stats['total']}")

    total_elapsed = time.perf_counter() - t0
    print(f"\nDone in {total_elapsed:.1f}s. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
