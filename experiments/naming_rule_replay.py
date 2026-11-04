#!/usr/bin/env python3
"""Score alternative corner-naming rules against a finished run (no GPU).

The tracker decides which of a pose's four cyclic namings to keep from the
keypoint correspondences (``flatsam2.match_naming``). That decision has exactly
one free number, ``tau``, plus whatever guard is layered on top, and trying each
combination as a grid would be absurd. So every run stores, per frame, the four
namings' residual histogram *and* the per-bin sum of squared residuals, which
makes the truncated cost

    sum_i min(d_i^2, tau^2)   =   sum_{bins below tau} sum_sq  +  tau^2 * count_above

**exact** for any ``tau`` that is a bin edge, recovered from the stored numbers
alone. This script turns that into a comparison of rules.

What it measures, and what it deliberately does not
---------------------------------------------------
It scores the *decision*, not the tracker: for every frame it asks which naming
ground truth would pick for the pose the branch actually produced, then asks
which naming each candidate rule picks. Accuracy on that question is exact and
needs no GPU.

It is **not** an end-to-end P@15 replay. Under ``corner_fusion_matchverify``
naming lives inside the verifier, so changing it changes which hypothesis the
branch returns and therefore the whole trajectory — the property
``fusion_rule_replay`` relies on does not hold here. Read this as "which rule
gets the naming right most often", then confirm the winner on one grid.

Rules compared
--------------
``tau=T``            argmax of the support at threshold T, retaining the deployed
                     eight-match structural minimum.
``tau=T,margin=K``   ... but keep the incumbent naming unless the winner's
                     support exceeds it by a factor K. Answers whether the guard
                     that the superseded distance-based rule needed (worth 0.73
                     pp there) is needed once the evidence is correspondences.
``tau=T,support=M``  ... and abstain below M effective matches.
``matches=M``        deployed tau with the raw-match minimum changed to M.
``incumbent``        never re-name: the floor any rule must beat.

**Read the coverage block before the rule table.** On a run that already names
inside the verifier (``register_support_power > 0``) the "incumbent" *is* the
post-naming pose, so every rule scores the same and the table says nothing —
by construction, not by accident. What remains open there is coverage: how many
localized frames carry no correspondence evidence at all, and how much of the
residual mis-naming lives in them. The rule table is informative only on a run
whose naming was *not* already decided this way.

Usage:
    python slurm/submit_cmd.py --gpus 0 --no-mail -- python experiments/naming_rule_replay.py \
        pti --config configs/tracker/corner_fusion_matchverify.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import pot_load_gt, pt_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import best_rotation_error
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

# Every bin edge is exact (flatsam2.match_naming.RESIDUAL_BIN_EDGES), so the
# sweep spans two orders of magnitude at no cost -- the point is to show the
# rule is insensitive across a wide band, not to find an argmax.
_TAUS = (3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)
_MARGINS = (1.0, 1.2, 1.5, 2.0)
_MIN_SUPPORT = (0.0, 4.0)
_MIN_MATCHES = (0, 4, 8, 12, 16, 32)
# Only frames whose pose is actually localized can have a *right* naming; a pose
# 200 px away is wrong under all four labelings and would only add noise.
_LOCALIZED_PX = 15.0
# flatsam2.config.SymmetryConfig.match_tau_px on the shipped config.
_DEPLOYED_TAU = 30.0
_DEPLOYED_MIN_MATCHES = 8


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    if loader == "pt":
        # PlanarTrack's own annotations -- the train split, where a parameter of
        # *our* method may be swept without selecting on the reported benchmark.
        return pt_load_gt
    raise ValueError(f"naming_rule_replay supports pot/pti/pt GT loaders, got {loader!r}")


def _deployed_pick(
    counts: np.ndarray,
    sum_sq: np.ndarray,
    edges: np.ndarray,
    n_matches: int,
) -> int:
    """The naming the shipped rule picks: argmax support at the deployed tau."""
    if n_matches < _DEPLOYED_MIN_MATCHES:
        return 0
    return int(np.argmax(support_at_tau(counts, sum_sq, edges, _DEPLOYED_TAU)))


def support_at_tau(
    counts: np.ndarray, sum_sq: np.ndarray, edges: np.ndarray, tau: float
) -> np.ndarray:
    """Effective-match support at ``tau``, recomputed from the stored histogram.

    Exact when ``tau`` is one of ``edges``; otherwise the bin straddling it is
    approximated by its own squared-sum, which is why the deployed edges include
    every threshold worth testing.

    Args:
        counts: (..., B) per-bin match counts.
        sum_sq: (..., B) per-bin sums of squared residuals.
        edges: (B + 1,) bin edges (px).
        tau: Truncation distance (px).

    Returns:
        (...,) support, ``sum max(0, 1 - d^2/tau^2)``.
    """
    upper = edges[1:]
    below = upper <= tau
    # sum over matches of (1 - d^2/tau^2), clipped at 0: bins entirely below tau
    # contribute exactly; bins entirely above contribute nothing.
    n_below = counts[..., below].sum(axis=-1)
    sq_below = sum_sq[..., below].sum(axis=-1)
    return n_below - sq_below / (tau**2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)
    sequences = get_sequences(
        args.dataset, base_dir, sequences=args.sequences, split=args.split
    )
    load_gt_fn = _load_gt_fn(args.dataset)

    rules: list[tuple[str, float, float, float]] = [("incumbent", 0.0, 0.0, 0.0)]
    for tau in _TAUS:
        for margin in _MARGINS:
            for min_support in _MIN_SUPPORT:
                label = f"tau={tau:g}"
                if margin > 1.0:
                    label += f",margin={margin:g}"
                if min_support > 0.0:
                    label += f",support={min_support:g}"
                rules.append((label, tau, margin, min_support))

    right = {label: 0 for label, *_ in rules}
    changed = {label: 0 for label, *_ in rules}
    min_match_right = {min_matches: 0 for min_matches in _MIN_MATCHES}
    min_match_changed = {min_matches: 0 for min_matches in _MIN_MATCHES}
    n_frames = 0
    n_seqs = 0
    # Coverage: a naming rule can only act where correspondences exist. On a run
    # that already names inside the verifier every rule scores the same, so the
    # question that is still open is how much of the data the evidence reaches.
    n_no_evidence = 0
    n_no_evidence_misnamed = 0
    n_with_evidence_misnamed = 0
    # Self-check: replaying the *deployed* rule must reproduce the run's own
    # `naming_shift`. If it does not, the replay is not modelling the tracker and
    # nothing below it means anything (same gate as inlier_threshold_sweep).
    n_deployed_agree = 0

    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            if "naming_hist_counts" not in data:
                continue
            n_seqs += 1
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            init_corners = gt_corners[0]
            frame_idx = data["frame_idx"]
            counts_all = data["naming_hist_counts"]
            sumsq_all = data["naming_hist_sumsq"]
            edges = data["naming_hist_edges"]
            n_matches = data["naming_n_matches"]
            samh_H_all = data["samh_H"]
            # The stored histograms are scored on the pose *before* naming, while
            # `samh_H` is the pose the branch returned *after* it. A rule's pick
            # therefore lives in a different frame of reference from `truth` by
            # exactly the shift the run applied, and comparing them directly
            # mis-scores every frame the run re-named. `naming_shift` is recorded
            # since 2026-08-19; without it the two frames coincide only when the
            # run never re-named, so refuse rather than report a wrong table.
            if "naming_shift" not in data:
                raise SystemExit(
                    f"{npz_path.name} predates the naming_shift telemetry; re-run the "
                    "sequences before replaying naming rules"
                )
            shift_all = data["naming_shift"]

            for t, fidx in enumerate(frame_idx):
                if fidx >= len(valid) or not valid[fidx]:
                    continue
                H = samh_H_all[t]
                if not np.all(np.isfinite(H)):
                    continue
                corners = warp_corners(H, init_corners)
                err, rot = best_rotation_error(gt_corners[fidx], corners)
                if float(err) > _LOCALIZED_PX:
                    continue
                truth = int(rot)
                if n_matches[t] < 0:
                    # Localized, so it has a right answer, but no correspondence
                    # evidence to find it with. This is where a naming mechanism
                    # built on matches structurally cannot help.
                    n_no_evidence += 1
                    n_no_evidence_misnamed += int(truth != 0)
                    continue
                n_with_evidence_misnamed += int(truth != 0)
                n_frames += 1
                applied = int(shift_all[t])
                applied = applied if applied > 0 else 0
                # A pick k (pre-naming frame) is right iff it lands on the same
                # naming as the run's own choice plus GT's correction.
                target = (applied + truth) % 4
                n_deployed_agree += int(
                    _deployed_pick(
                        counts_all[t], sumsq_all[t], edges, int(n_matches[t])
                    )
                    == applied
                )
                for label, tau, margin, min_support in rules:
                    if label == "incumbent":
                        # "never re-name" means keeping what the branch produced
                        # before naming, i.e. shift 0 in the pre-naming frame.
                        pick = 0
                    else:
                        support = support_at_tau(counts_all[t], sumsq_all[t], edges, tau)
                        best = int(np.argmax(support))
                        pick = best
                        if n_matches[t] < _DEPLOYED_MIN_MATCHES:
                            pick = 0
                        elif support[best] < min_support:
                            pick = 0
                        elif best != 0 and support[best] < margin * support[0]:
                            pick = 0
                    right[label] += int(pick == target)
                    changed[label] += int(pick != 0)

                support = support_at_tau(
                    counts_all[t], sumsq_all[t], edges, _DEPLOYED_TAU
                )
                best = int(np.argmax(support))
                for min_matches in _MIN_MATCHES:
                    pick = 0 if n_matches[t] < min_matches else best
                    min_match_right[min_matches] += int(pick == target)
                    min_match_changed[min_matches] += int(pick != 0)

    if n_seqs == 0:
        print("No naming telemetry found — run a config with register_support_power > 0.")
        return 1

    total_localized = n_frames + n_no_evidence
    print(f"config={args.config.stem}  dataset={ds_info.label}  sequences={n_seqs}")
    print(f"localized GT frames: {total_localized}")
    print(
        f"  with correspondence evidence: {n_frames} "
        f"({100.0 * n_frames / max(total_localized, 1):.1f}%), "
        f"mis-named {n_with_evidence_misnamed} "
        f"({100.0 * n_with_evidence_misnamed / max(n_frames, 1):.2f}%)"
    )
    print(
        f"  without evidence:             {n_no_evidence} "
        f"({100.0 * n_no_evidence / max(total_localized, 1):.1f}%), "
        f"mis-named {n_no_evidence_misnamed} "
        f"({100.0 * n_no_evidence_misnamed / max(n_no_evidence, 1):.2f}%)"
    )
    print(
        "  -> of all mis-named localized frames, "
        f"{100.0 * n_no_evidence_misnamed / max(n_no_evidence_misnamed + n_with_evidence_misnamed, 1):.1f}%"
        " are beyond any correspondence-based rule's reach\n"
    )
    agree_pct = 100.0 * n_deployed_agree / max(n_frames, 1)
    verdict = "MATCH" if agree_pct >= 99.99 else "MISMATCH"
    print(
        f"  self-check: replaying the deployed rule (tau={_DEPLOYED_TAU:g}) reproduces the "
        f"run's own naming on {agree_pct:.2f}% of frames -> {verdict}"
    )
    if verdict == "MISMATCH":
        print("  ** the replay does not model the tracker; the table below is not evidence **")
    print()
    print("  rule                       | naming correct | re-named")
    print("  " + "-" * 56)
    for label, *_ in rules:
        pct = 100.0 * right[label] / n_frames if n_frames else float("nan")
        chg = 100.0 * changed[label] / n_frames if n_frames else float("nan")
        print(f"  {label:<26} | {pct:13.2f}% | {chg:7.2f}%")
    print()
    print("  raw-match minimum | naming correct | re-named")
    print("  " + "-" * 48)
    for min_matches in _MIN_MATCHES:
        pct = (
            100.0 * min_match_right[min_matches] / n_frames
            if n_frames
            else float("nan")
        )
        chg = (
            100.0 * min_match_changed[min_matches] / n_frames
            if n_frames
            else float("nan")
        )
        deployed = " (deployed)" if min_matches == _DEPLOYED_MIN_MATCHES else ""
        print(f"  {min_matches:>17}{deployed:<11} | {pct:13.2f}% | {chg:7.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
