"""Sanity-check the POIC / TMT / UCSB / MPOT-3K loaders against the data on disk.

Lists every sequence of each dataset, loads its ground truth and frame paths,
and reports the shapes, the annotated fraction, and anything that fails.  This
is the cheap CPU-only check that the new dataset wiring is right before any GPU
time is spent tracking on it.

Run:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/check_new_datasets.py
"""

import argparse
import traceback
from pathlib import Path

import numpy as np

from flatsam2.datasets import DATASETS, get_sequences, make_gt_loader, make_image_paths
from flatsam2.path_config import paths


def check_dataset(dataset: str, split: str, max_sequences: int) -> int:
    """Load every sequence of *dataset* and print a one-line summary per failure.

    Args:
        dataset: Dataset CLI key.
        split: Split name, used only by datasets that have splits.
        max_sequences: Check at most this many sequences (0 = all).

    Returns:
        Number of sequences that failed to load.
    """
    ds_info = DATASETS[dataset]
    base_dir = getattr(paths, ds_info.base_dir_attr)
    load_gt = make_gt_loader(dataset)
    image_paths_of = make_image_paths(dataset)

    print(f"\n=== {ds_info.label}  ({base_dir}) ===")
    sequences = get_sequences(dataset, base_dir, split=split if ds_info.uses_split else None)
    if max_sequences:
        sequences = sequences[:max_sequences]
    print(f"{len(sequences)} sequences/tasks")

    n_failed = 0
    total_frames = 0
    annotated_fractions = []
    for seq_name in sequences:
        try:
            annot, valid = load_gt(base_dir, seq_name)
            image_paths = image_paths_of(base_dir, seq_name)
            if annot.shape != (len(image_paths), 2, 4):
                raise ValueError(
                    f"shape mismatch: annot {annot.shape} vs {len(image_paths)} frames"
                )
            if not valid[0]:
                raise ValueError("first frame is not annotated")
            total_frames += len(image_paths)
            annotated_fractions.append(float(valid.mean()))
        except Exception as exc:  # noqa: BLE001 - a report, not a control path
            n_failed += 1
            print(f"  FAIL {seq_name}: {exc}")

    if annotated_fractions:
        fractions = np.array(annotated_fractions)
        print(
            f"  ok {len(fractions)}/{len(sequences)}   frames {total_frames}   "
            f"annotated: mean {100 * fractions.mean():.1f}%  min {100 * fractions.min():.1f}%"
        )
        partial = [s for s, f in zip(sequences, annotated_fractions) if f < 1.0]
        if partial:
            print(f"  partially annotated ({len(partial)}): {', '.join(partial[:6])}")

    # Spot-check the first sequence so corner ordering and scale are visible.
    if sequences and n_failed < len(sequences):
        annot, valid = load_gt(base_dir, sequences[0])
        image_paths = image_paths_of(base_dir, sequences[0])
        print(f"  first task {sequences[0]!r}: {len(image_paths)} frames, "
              f"first frame {image_paths[0].name}")
        print(f"    init corners x={np.round(annot[0][0], 1)} y={np.round(annot[0][1], 1)}")

    return n_failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["poic", "tmt", "ucsb", "mpot"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-sequences", type=int, default=0,
                        help="check at most this many sequences per dataset (0 = all)")
    args = parser.parse_args()

    total_failed = 0
    for dataset in args.datasets:
        try:
            total_failed += check_dataset(dataset, args.split, args.max_sequences)
        except Exception:  # noqa: BLE001 - keep going through the other datasets
            total_failed += 1
            print(f"\n=== {dataset}: dataset-level failure ===")
            traceback.print_exc()

    print(f"\n{'FAILURES: ' + str(total_failed) if total_failed else 'all datasets loaded cleanly'}")


if __name__ == "__main__":
    main()
