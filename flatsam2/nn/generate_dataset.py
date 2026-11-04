"""Pre-generate a synthetic training dataset and save to disk.

Running data generation on-the-fly during training is CPU-bound and limits
GPU throughput.  This script generates a fixed dataset once (ideally on
CPU-only cluster nodes), saving numpy arrays that the training job can read
at disk speed.

Usage
-----
    # Generate 500 K samples for option_e, 32 parallel workers:
    python -m flatsam2.nn.generate_dataset \\
        --output-dir /scratch/datasets/nn_quality_500k \\
        --n-samples 500000 \\
        --num-workers 32

    # Generate for the point_baseline (no SDF — faster):
    python -m flatsam2.nn.generate_dataset \\
        --arch point_baseline \\
        --output-dir /scratch/datasets/nn_quality_baseline_500k \\
        --n-samples 500000 \\
        --num-workers 32

    # Then pass to training:
    python -m flatsam2.nn.train \\
        --arch option_e \\
        --cache-dir /scratch/datasets/nn_quality_500k \\
        --output-dir checkpoints/option_e

Output layout
-------------
    option_e / option_c / both:
        features_A.npy   (N, 128, 20) float32
        features_B.npy   (N, 128, 20) float32

    point_baseline:
        features_combined.npy  (N, 256, 18) float32

    all architectures:
        gt_rmse.npy      (N,)         float32
        censored.npy     (N,)         bool
        meta.json        generation config + stats
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_WORKER_REPORT_INTERVAL_S = 60.0  # print progress once per minute per worker


def _generate_chunk(
    n_target: int,
    seed: int,
    architecture: str,
    n_points: int,
    worker_id: int = 0,
) -> dict:
    """Generate *n_target* valid samples in a single worker process.

    Returns a dict of stacked arrays ready to be concatenated by the caller.
    """
    # Import inside worker so each subprocess gets a clean state.
    from flatsam2.nn.data_gen import generate_sample

    rng = np.random.default_rng(seed)
    feat_A_list: List[np.ndarray] = []
    feat_B_list: List[np.ndarray] = []
    feat_combined_list: List[np.ndarray] = []
    gt_rmse_list: List[float] = []
    censored_list: List[bool] = []
    is_baseline = (architecture == "point_baseline")
    attempts = 0
    t0 = time.perf_counter()
    last_report = t0
    print(f"  worker {worker_id:02d}: starting  target={n_target}", flush=True)

    counter = feat_combined_list if is_baseline else feat_A_list
    while len(counter) < n_target:
        sample = generate_sample(rng, n_points=n_points, architecture=architecture)
        attempts += 1
        if sample is None:
            continue
        if is_baseline:
            feat_combined_list.append(sample["features_combined"])
        else:
            feat_A_list.append(sample["features_A"])
            feat_B_list.append(sample["features_B"])
        gt_rmse_list.append(sample["gt_rmse"])
        censored_list.append(sample["censored"])

        now = time.perf_counter()
        if now - last_report >= _WORKER_REPORT_INTERVAL_S:
            done = len(counter)
            elapsed = now - t0
            rate = done / elapsed
            eta = (n_target - done) / max(rate, 1e-6)
            print(
                f"  worker {worker_id:02d}: {done}/{n_target} ({100*done/n_target:.0f}%)"
                f"  {rate:.1f} smp/s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                flush=True,
            )
            last_report = now

    elapsed = time.perf_counter() - t0
    print(
        f"  worker {worker_id:02d}: done  {n_target} samples in {elapsed/60:.1f}m"
        f"  ({n_target/elapsed:.1f} smp/s)",
        flush=True,
    )
    result: dict = {
        "gt_rmse":  np.array(gt_rmse_list, dtype=np.float32),
        "censored": np.array(censored_list, dtype=bool),
        "attempts": attempts,
    }
    if is_baseline:
        result["features_combined"] = np.stack(feat_combined_list).astype(np.float32)
    else:
        result["features_A"] = np.stack(feat_A_list).astype(np.float32)
        result["features_B"] = np.stack(feat_B_list).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

_CHUNKS_SUBDIR = "chunks"


def _chunk_path(out_dir: Path, chunk_id: int) -> Path:
    return out_dir / _CHUNKS_SUBDIR / f"chunk_{chunk_id:06d}.npz"


def _scan_chunks(out_dir: Path) -> tuple:
    """Return (n_chunk_files, n_samples_in_chunks) from chunks/ subdir."""
    d = out_dir / _CHUNKS_SUBDIR
    if not d.exists():
        return 0, 0
    n_files, n_samples = 0, 0
    for p in sorted(d.glob("chunk_*.npz")):
        try:
            with np.load(p) as data:
                n_samples += int(data["gt_rmse"].shape[0])
                n_files += 1
        except Exception:
            pass
    return n_files, n_samples


def _count_merged(out_dir: Path) -> int:
    f = out_dir / "gt_rmse.npy"
    if not f.exists():
        return 0
    try:
        return int(np.load(f, mmap_mode="r").shape[0])
    except Exception:
        return 0


def _merge_chunks(out_dir: Path, architecture: str = "option_e") -> int:
    """Merge chunks/ into the final .npy files; delete chunk files on success.

    Returns the total number of samples after merging.
    """
    chunks_dir = out_dir / _CHUNKS_SUBDIR
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz")) if chunks_dir.exists() else []
    if not chunk_files:
        return _count_merged(out_dir)

    is_baseline = (architecture == "point_baseline")
    feat_keys = ["features_combined"] if is_baseline else ["features_A", "features_B"]

    print(f"Merging {len(chunk_files)} chunk files...", flush=True)
    parts: dict = {k: [] for k in feat_keys + ["gt_rmse", "censored"]}

    # Prepend existing merged data if present
    if (out_dir / "gt_rmse.npy").exists():
        for k in feat_keys:
            parts[k].append(np.load(out_dir / f"{k}.npy"))
        parts["gt_rmse"].append(np.load(out_dir / "gt_rmse.npy"))
        parts["censored"].append(np.load(out_dir / "censored.npy"))

    for p in chunk_files:
        with np.load(p) as data:
            for k in parts:
                parts[k].append(data[k])

    for k in feat_keys:
        np.save(out_dir / f"{k}.npy", np.concatenate(parts[k], axis=0))
    np.save(out_dir / "gt_rmse.npy",  np.concatenate(parts["gt_rmse"],  axis=0))
    np.save(out_dir / "censored.npy", np.concatenate(parts["censored"], axis=0))

    # Only delete chunks after successful save
    for p in chunk_files:
        p.unlink()
    try:
        chunks_dir.rmdir()  # removes dir if now empty
    except OSError:
        pass

    total = _count_merged(out_dir)
    print(f"Merge complete: {total:,} total samples.", flush=True)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-generate synthetic NN training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write dataset files into.")
    parser.add_argument("--n-samples", type=int, default=500_000,
                        help="Target total number of samples (resumes if some already exist).")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of parallel worker processes.")
    parser.add_argument("--arch", default="option_e",
                        choices=["option_e", "option_c", "both", "point_baseline"])
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; worker i uses seed+i.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _CHUNKS_SUBDIR).mkdir(exist_ok=True)

    n_existing_chunks, chunk_samples = _scan_chunks(out_dir)
    merged_samples = _count_merged(out_dir)
    already = merged_samples + chunk_samples
    n_remaining = max(0, args.n_samples - already)

    if n_remaining == 0:
        print(f"Already have {already:,} samples >= target {args.n_samples:,}.")
        if chunk_samples > 0:
            _merge_chunks(out_dir, architecture=args.arch)
        return

    if already > 0:
        print(f"Resuming: {already:,} samples exist ({merged_samples:,} merged + "
              f"{chunk_samples:,} in {n_existing_chunks} chunk files), "
              f"need {n_remaining:,} more to reach {args.n_samples:,}.")
    else:
        print(f"Generating {n_remaining:,} samples  arch={args.arch}  "
              f"workers={args.num_workers}  seed={args.seed}")
    print(f"Output: {out_dir}")

    n_workers = args.num_workers
    chunk_size = max(1, n_remaining // n_workers)
    chunk_sizes = [chunk_size] * n_workers
    chunk_sizes[-1] += n_remaining - chunk_size * n_workers
    # Offset seeds so resumed runs never repeat earlier seeds.
    seed_offset = args.seed + already

    total_attempts = 0
    done = 0
    t0 = time.perf_counter()

    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, n in enumerate(chunk_sizes):
            fut = pool.submit(
                _generate_chunk, n, seed_offset + i, args.arch, args.n_points, i)
            futures[fut] = i

        for fut in as_completed(futures):
            chunk = fut.result()
            n_chunk = len(chunk["gt_rmse"])

            # Save this chunk to disk immediately — survives job failure.
            chunk_id = n_existing_chunks + futures[fut]
            save_arrays: dict = {
                "gt_rmse": chunk["gt_rmse"],
                "censored": chunk["censored"],
            }
            if args.arch == "point_baseline":
                save_arrays["features_combined"] = chunk["features_combined"]
            else:
                save_arrays["features_A"] = chunk["features_A"]
                save_arrays["features_B"] = chunk["features_B"]
            np.savez(_chunk_path(out_dir, chunk_id), **save_arrays)

            total_attempts += chunk["attempts"]
            done += n_chunk
            elapsed = time.perf_counter() - t0
            sps = done / elapsed
            eta = (n_remaining - done) / max(sps, 1e-6)
            workers_done = sum(1 for f in futures if f.done())
            print(
                f"  chunk saved ({workers_done}/{n_workers} workers done)"
                f"  total {already + done:>8,}/{args.n_samples:,}"
                f"  {sps:.1f} smp/s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"\nAll workers done: {done:,} new samples in {elapsed/60:.1f}m"
          f"  ({done/elapsed:.1f} smp/s)"
          f"  success rate {done/total_attempts:.1%}")

    total = _merge_chunks(out_dir, architecture=args.arch)

    meta = {
        "n_samples": total,
        "arch": args.arch,
        "n_points": args.n_points,
        "seed": args.seed,
        "generation_time_s": round(elapsed, 1),
        "success_rate": round(done / total_attempts, 4),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. {total:,} samples in {out_dir}")


if __name__ == "__main__":
    main()
