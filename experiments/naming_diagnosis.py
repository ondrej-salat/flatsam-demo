#!/usr/bin/env python3
"""Can the keypoint branch name the mask branch's corners? (offline, no GPU)

The whole remaining PTi deficit of the shipped tracker is corner *naming*:
`br_p@15` 90.24 against `p@15` 88.32, i.e. on ~2% of GT frames the quad is in
the right place with the corners labelled in the wrong cyclic order. A
silhouette cannot fix that (docs/state-of-the-tracker.md section 4), but the
keypoint branch matches texture and therefore can.

This script asks, on a finished run's stored per-frame branch estimates, three
questions that decide whether wiring LoMa's naming into the mask branch is
worth anything:

1. **How much is on the table** — the mis-named frames (as-is error > 15 px,
   best-rotation error <= 15 px), and the fusion mode that produced them.
2. **Does LoMa know the naming** — on those frames, is LoMa's own quad named
   correctly (its as-is rotation is the one GT picks)?
3. **Would the obvious rule fire correctly** — relabel the SAM-H quad to the
   cyclic rotation nearest LoMa's quad; does that rotation equal the one GT
   would choose? Counted both ways: fixes on currently-wrong frames, and
   breakages on currently-right ones.

Nothing here is a simulation: every number is a readout of estimates the
deployed tracker actually produced. Whether the rule *helps end to end* is a
separate question, answered exactly by `experiments/fusion_rule_replay.py`,
because a naming step at fusion time only re-labels stored branch estimates.

Usage:
    python slurm/submit_cmd.py --gpus 0 --no-mail -- python experiments/naming_diagnosis.py \
        pti --config configs/tracker/corner_fusion_wholeframe.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import pot_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error, best_rotation_error
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

_P_AT_K_PX = 15.0


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    raise ValueError(f"naming_diagnosis supports pot/pti GT loaders, got {loader!r}")


def _finite_H(H: np.ndarray) -> np.ndarray | None:
    """The homography, or None when the run recorded it as absent (NaN)."""
    return H if np.all(np.isfinite(H)) else None


def _rotation_nearest(quad: np.ndarray, reference: np.ndarray) -> int:
    """Cyclic rotation of ``quad`` whose corners sit closest to ``reference``.

    Args:
        quad: (2, 4) corner positions to be relabelled.
        reference: (2, 4) corner positions whose naming is trusted.

    Returns:
        Shift r in 0..3 minimizing ``||roll(quad, r) - reference||_F``.
    """
    return int(np.argmin([
        float(np.linalg.norm(np.roll(quad, r, axis=1) - reference)) for r in range(4)
    ]))


def _rotation_gt(quad: np.ndarray, gt: np.ndarray) -> tuple[int, float]:
    """Cyclic rotation of ``quad`` GT would pick, and the error it achieves."""
    err, rot = best_rotation_error(gt, quad)
    return int(rot), float(err)


def diagnose(args: argparse.Namespace) -> int:
    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)
    sequences = get_sequences(
        args.dataset, base_dir, sequences=args.sequences, subset=args.subset, split=args.split
    )
    load_gt_fn = _load_gt_fn(args.dataset)

    n_seqs = 0
    n_gt = 0
    misnamed_by_mode: Counter[str] = Counter()
    seqs_with_misnaming: Counter[str] = Counter()

    # Question 2/3 counters, restricted to frames where both branches exist.
    loma_named_right = 0          # on mis-named frames: LoMa's own naming is GT's
    loma_localized = 0            # on mis-named frames: LoMa's quad is within 15 px br
    rule_would_fix = 0            # mis-named frame, nearest-to-LoMa rotation == GT's
    rule_wrong_pick = 0           # mis-named frame, rule picks a different rotation
    rule_no_loma = 0              # mis-named frame with no LoMa pose to name from
    ok_frames_both = 0            # currently-correct frames with both branches
    rule_would_break = 0          # ... where the rule would relabel the SAM-H quad away
    samh_misnamed = 0             # SAM-H branch itself mis-named (pose ok, naming not)
    samh_naming_recoverable = 0   # ... and nearest-to-LoMa rotation is GT's

    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        pred_path = seq_dir / f"{seq_name}.txt"
        if not (npz_path.exists() and pred_path.exists()):
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            if "fusion_samh_cov" not in data:
                continue
            n_seqs += 1
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            init_corners = gt_corners[0]
            pred = np.loadtxt(pred_path).reshape(-1, 4, 2).transpose(0, 2, 1)
            frame_idx = data["frame_idx"]
            modes = data["fusion_modes"]
            loma_H_all, samh_H_all = data["loma_H"], data["samh_H"]

            for t, fidx in enumerate(frame_idx):
                if fidx >= len(valid) or not valid[fidx] or fidx >= len(pred):
                    continue
                n_gt += 1
                gt = gt_corners[fidx]
                out = pred[fidx]
                err = float(alignment_error(gt, out))
                rot_gt_out, br_err = _rotation_gt(out, gt)
                loma_H = _finite_H(loma_H_all[t])
                samh_H = _finite_H(samh_H_all[t])
                c_loma = warp_corners(loma_H, init_corners) if loma_H is not None else None
                c_samh = warp_corners(samh_H, init_corners) if samh_H is not None else None

                misnamed = err > _P_AT_K_PX >= br_err
                if misnamed:
                    misnamed_by_mode[str(modes[t])] += 1
                    seqs_with_misnaming[seq_name] += 1
                    if c_loma is None:
                        rule_no_loma += 1
                    else:
                        rot_gt_loma, br_loma = _rotation_gt(c_loma, gt)
                        loma_named_right += int(rot_gt_loma == 0)
                        loma_localized += int(br_loma <= _P_AT_K_PX)
                        # The rule names the *output* quad from LoMa's naming.
                        if _rotation_nearest(out, c_loma) == rot_gt_out:
                            rule_would_fix += 1
                        else:
                            rule_wrong_pick += 1
                elif err <= _P_AT_K_PX and c_loma is not None and c_samh is not None:
                    ok_frames_both += 1
                    rule_would_break += int(_rotation_nearest(out, c_loma) != 0)

                if c_samh is not None and c_loma is not None:
                    err_samh = float(alignment_error(gt, c_samh))
                    rot_gt_samh, br_samh = _rotation_gt(c_samh, gt)
                    if err_samh > _P_AT_K_PX >= br_samh:
                        samh_misnamed += 1
                        samh_naming_recoverable += int(
                            _rotation_nearest(c_samh, c_loma) == rot_gt_samh
                        )

    if n_seqs == 0:
        print("No corner_fusion debug npz found — nothing to diagnose.")
        return 1

    n_misnamed = sum(misnamed_by_mode.values())
    print(f"config={args.config.stem}  dataset={ds_info.label}  sequences={n_seqs}  "
          f"GT frames={n_gt}")
    print(f"\n1. Mis-named output frames (err > 15, best-rotation <= 15): "
          f"{n_misnamed} ({100.0 * n_misnamed / n_gt:.2f}% of GT frames), "
          f"on {len(seqs_with_misnaming)}/{n_seqs} sequences")
    print("   by fusion mode:")
    for mode, count in misnamed_by_mode.most_common():
        print(f"     {mode:<16} {count:6d}  ({100.0 * count / n_misnamed:5.1f}%)")
    print("   worst sequences:")
    for seq, count in seqs_with_misnaming.most_common(10):
        print(f"     {seq:<16} {count:6d}")

    with_loma = n_misnamed - rule_no_loma
    print(f"\n2. On those frames, LoMa has a pose on {with_loma}/{n_misnamed}")
    if with_loma:
        print(f"     LoMa's own naming is GT's:      {loma_named_right:6d} "
              f"({100.0 * loma_named_right / with_loma:5.1f}%)")
        print(f"     LoMa's quad localized (br<=15): {loma_localized:6d} "
              f"({100.0 * loma_localized / with_loma:5.1f}%)")
    print(f"\n3. Rule 'relabel to the rotation nearest LoMa':")
    if with_loma:
        print(f"     would fix:        {rule_would_fix:6d} "
              f"({100.0 * rule_would_fix / with_loma:5.1f}% of mis-named-with-LoMa)")
        print(f"     picks wrong rot:  {rule_wrong_pick:6d}")
    print(f"     would break:      {rule_would_break:6d} of {ok_frames_both} currently-OK "
          f"frames ({100.0 * rule_would_break / max(ok_frames_both, 1):.2f}%)")
    print(f"\n   SAM-H branch mis-named (pose ok, naming not): {samh_misnamed} frames; "
          f"nearest-to-LoMa recovers {samh_naming_recoverable} "
          f"({100.0 * samh_naming_recoverable / max(samh_misnamed, 1):.1f}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    return diagnose(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
