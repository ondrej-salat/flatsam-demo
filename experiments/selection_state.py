#!/usr/bin/env python3
"""Stateful candidate selection (S1 v3): breakdowns are episodes, not frames.

Why this exists
---------------
The stateless selector study (``selector_study.py``) is exhausted: threshold
stumps over every online feature recover +0.20 pp held-out P@15 against a
+3.0 pp ceiling, and the two-sided smoothness arbitration self-neutered
(optimal tau -> infinity).  The manual failure-type tags on the breakdown
gallery say why.  Two thirds of the tagged high-inlier catastrophes are
*identity* failures (LoMA locked onto a distractor / a same-looking part / a
rotated alias); they separate cleanly at ``disagreement > 60 px``, and they are
not single-frame events — most of their frame mass sits in runs tens of frames
long.  A per-frame arbiter that compares the two branches' temporal smoothness
can only work on the *onset* frame: once the tracker has been sitting on the
distractor for ten frames, both branches move smoothly again and the signal is
gone.  So the arbiter had to be right on every frame of an episode while its
evidence existed on only the first one — hence the self-neutering.

This script tests the structural fix: **detect the onset, then latch**.  A
two-state selector (trust LoMA / distrust LoMA) enters the distrust state when
the branches diverge and the LoMA quad is the one that jumped, forces the SAM-H
candidate while there, and leaves only after the branches agree again for a few
frames.  One correct detection then rescues the whole episode.

What it reports
---------------
1. Episode structure — run-length distribution of LoMA-bad frames, and how much
   of the rescuable mass sits in long runs (the size of the prize).
2. Onset vs sustained feature AUCs — the diagnosis above, as numbers.
3. Rule frontier — baseline, per-frame-oracle ceiling, stateless disagreement
   stump, the latch family (with and without the onset speed arbiter), and two
   oracle latches that bound what perfect episode detection / perfect
   per-episode decisions could buy.
4. Failure-type tag join (when ``loma_tags.json`` is present) — per-tag
   populations and how many tagged frames each rule actually rescues.

Every table is reported at P@5, P@15 and P@30.  P@15 stays the decision metric;
P@30 is the GT-noise control (PlanarTrack annotations average ~5.7 px error and
~10% exceed 15 px, so a rule that only trades annotation noise moves P@15 and
P@30 differently than one that fixes real breakdowns).

Substrate: the ``selector_study.py dump`` per-sequence CSVs (E0+ dry run).  No
GPU, no mask IO, no dataset access — login-node friendly.  Run it with no
arguments:

    python experiments/selection_state.py

Caveat (shared with ``selector_study.py``): re-selection is simulated over
*stored* per-frame candidates, so it cannot model how a different choice would
have changed the tracker's own downstream state.  It bounds a re-ranking of
existing candidates, not a re-run.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from flatsam2.path_config import paths
from flatsam2.registry import DATASETS


_LABEL = DATASETS["pti"].label
_PER_SEQ_DIR = paths.OUTPUT_DIRECTORY / "experiments" / "selector_study" / _LABEL / "per_sequence"
_TAGS_PATH = paths.OUTPUT_DIRECTORY / "experiments" / "loma_breakdowns" / _LABEL / "loma_tags.json"
_OUTPUT_DIR = paths.OUTPUT_DIRECTORY / "experiments" / "selection_state" / _LABEL

_FEATURES = (
    "n_inliers",
    "disagreement",
    "spill_loma",
    "excess_loma",
    "area_log_ratio",
    "loma_speed",
    "samh_speed",
    "speed_gap",
)

# A frame counts as a LoMA breakdown above this error; the same 15 px the
# decision metric uses, so "bad" here means "a P@15 miss the branch caused".
_BAD_PX = 15.0
# Candidate-comparison margin: below it the two candidates are indistinguishable
# at the PlanarTrack annotation-noise floor.
_LABEL_MARGIN = 3.0
_THRESHOLDS = (5.0, 15.0, 30.0)

# Latch grids. tau_hi enters the distrust state, tau_lo (as a fraction of
# tau_hi) leaves it, k_exit frames of agreement are required to leave.
_TAU_HI_GRID = (20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 200.0)
_TAU_LO_FRAC_GRID = (0.25, 0.5, 1.0)
_K_EXIT_GRID = (1, 3, 10)


@dataclass
class SeqFrames:
    """One sequence's per-frame selector substrate, in stored frame order."""

    seq: str
    frame_idx: np.ndarray          # (N,) int
    evaluated: np.ndarray          # (N,) bool — has ground truth
    baseline_pick: np.ndarray      # (N,) bool — True = the run chose samh_H
    err_loma: np.ndarray           # (N,) float px, NaN when unavailable
    err_samh: np.ndarray           # (N,) float px, NaN when unavailable
    feats: dict[str, np.ndarray]   # feature name -> (N,) float


# --------------------------------------------------------------------------- #
# substrate
# --------------------------------------------------------------------------- #


def load_sequences(per_seq_dir: Path) -> list[SeqFrames]:
    """Read the selector-study per-sequence CSVs, keeping per-sequence order.

    Args:
        per_seq_dir: Directory of ``{seq}.csv`` files written by
            ``selector_study.py dump``.

    Returns:
        One ``SeqFrames`` per non-empty CSV, sorted by sequence name.

    Raises:
        FileNotFoundError: When the directory holds no CSVs.
        ValueError: When a CSV predates the v2 speed columns.
    """
    files = sorted(per_seq_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no selector_study CSVs in {per_seq_dir} — run "
            "`python experiments/selector_study.py dump` first"
        )
    out: list[SeqFrames] = []
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
        if not raw:
            continue
        if "loma_speed" not in raw[0]:
            raise ValueError(
                f"{path} lacks the speed columns — re-run "
                "`selector_study.py dump --no-skip-completed`"
            )
        col = lambda name: np.asarray([float(d[name]) for d in raw])  # noqa: E731
        out.append(
            SeqFrames(
                seq=path.stem,
                frame_idx=col("frame_idx").astype(int),
                evaluated=col("evaluated") == 1,
                baseline_pick=np.asarray([d["selected"] == "samh_H" for d in raw]),
                err_loma=col("err_loma"),
                err_samh=col("err_samh"),
                feats={name: col(name) for name in _FEATURES},
            )
        )
    return out


def train_test_split(seqs: list[SeqFrames]) -> tuple[set[str], set[str]]:
    """Deterministic sequence-level split (even index train, odd test).

    Matches ``selector_study.py`` so the two studies' held-out numbers compare.
    """
    names = sorted(sf.seq for sf in seqs)
    return set(names[0::2]), set(names[1::2])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def rule_errors(sf: SeqFrames, pick_samh: np.ndarray) -> np.ndarray:
    """Per-frame error of a selection, falling back to the other candidate."""
    err = np.where(pick_samh, sf.err_samh, sf.err_loma)
    return np.where(np.isfinite(err), err, np.where(pick_samh, sf.err_loma, sf.err_samh))


def precisions(errs: np.ndarray) -> dict[str, float]:
    """P@5 / P@15 / P@30 over the finite entries of ``errs`` (percent)."""
    finite = errs[np.isfinite(errs)]
    if len(finite) == 0:
        return {f"p{int(t)}": float("nan") for t in _THRESHOLDS}
    return {f"p{int(t)}": float(100.0 * np.mean(finite <= t)) for t in _THRESHOLDS}


def evaluate_rule(
    seqs: list[SeqFrames],
    picks: dict[str, np.ndarray],
    keep_seqs: set[str],
) -> tuple[dict[str, float], float]:
    """Precisions over the evaluated frames of ``keep_seqs``, plus switch rate.

    Args:
        seqs: All sequences.
        picks: seq name -> (N,) bool selection.
        keep_seqs: Sequence names to score (the train or test half).

    Returns:
        (precision dict, fraction of frames where the rule differs from the run).
    """
    errs: list[np.ndarray] = []
    changed = total = 0
    for sf in seqs:
        if sf.seq not in keep_seqs:
            continue
        pick = picks[sf.seq]
        errs.append(rule_errors(sf, pick)[sf.evaluated])
        changed += int((pick != sf.baseline_pick).sum())
        total += len(pick)
    return precisions(np.concatenate(errs) if errs else np.zeros(0)), (
        changed / total if total else 0.0
    )


def oracle_picks(seqs: list[SeqFrames]) -> dict[str, np.ndarray]:
    """Per-frame oracle: take whichever candidate has the smaller error."""
    return {
        sf.seq: np.where(
            np.isfinite(sf.err_samh) & np.isfinite(sf.err_loma),
            sf.err_samh < sf.err_loma,
            ~np.isfinite(sf.err_loma),
        )
        for sf in seqs
    }


# --------------------------------------------------------------------------- #
# episode structure
# --------------------------------------------------------------------------- #


def bad_runs(sf: SeqFrames) -> list[tuple[int, int]]:
    """Maximal runs of consecutive LoMA-bad frames as ``(start, stop)`` slices.

    "LoMA-bad" = evaluated, LoMA error finite and above ``_BAD_PX``.  Runs break
    across gaps in ``frame_idx`` (the dry run stores frames sparsely on some
    sequences), so a run is genuinely contiguous in time.
    """
    bad = sf.evaluated & np.isfinite(sf.err_loma) & (sf.err_loma > _BAD_PX)
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for k in range(len(bad)):
        contiguous = k > 0 and sf.frame_idx[k] == sf.frame_idx[k - 1] + 1
        if bad[k] and (start is None or not contiguous):
            if start is not None:
                runs.append((start, k))
            start = k
        elif not bad[k] and start is not None:
            runs.append((start, k))
            start = None
    if start is not None:
        runs.append((start, len(bad)))
    return runs


def episode_report(seqs: list[SeqFrames]) -> tuple[list[list], dict[str, float]]:
    """Run-length structure of LoMA breakdowns, overall and for rescuable mass."""
    rows: list[list] = []
    summary: dict[str, float] = {}
    for name, rescuable_only in (("all LoMA-bad", False), ("rescuable (samh <= 15px)", True)):
        lengths: list[int] = []
        n_frames = 0
        for sf in seqs:
            for a, b in bad_runs(sf):
                sel = slice(a, b)
                if rescuable_only:
                    good = np.isfinite(sf.err_samh[sel]) & (sf.err_samh[sel] <= _BAD_PX)
                    if not good.any():
                        continue
                    n_frames += int(good.sum())
                else:
                    n_frames += b - a
                lengths.append(b - a)
        if not lengths:
            continue
        arr = np.asarray(lengths)
        frac10 = float(arr[arr >= 10].sum() / arr.sum())
        rows.append(
            [
                name,
                n_frames,
                len(arr),
                f"{np.median(arr):.0f}",
                f"{arr.mean():.1f}",
                int(arr.max()),
                f"{100 * float(arr[arr >= 5].sum() / arr.sum()):.0f}%",
                f"{100 * frac10:.0f}%",
            ]
        )
        key = "rescuable" if rescuable_only else "all"
        summary[f"episodes_{key}"] = len(arr)
        summary[f"frames_{key}"] = n_frames
        summary[f"frac_in_runs_ge10_{key}"] = frac10
    return rows, summary


# --------------------------------------------------------------------------- #
# onset vs sustained diagnosis
# --------------------------------------------------------------------------- #


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC of ``scores`` for binary ``labels`` (midranks for ties)."""
    if len(scores) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def onset_sustained_aucs(seqs: list[SeqFrames], train: set[str]) -> list[list]:
    """Feature AUCs (label: samh clearly better) split by onset vs sustained.

    Onset = first frame of a LoMA-bad run; sustained = the rest of the run.  A
    feature that only fires at onset carries no information on the frames where
    most of the metric mass is.
    """
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        pop: {"y": [], **{f: [] for f in _FEATURES}} for pop in ("onset", "sustained")
    }
    for sf in seqs:
        if sf.seq not in train:
            continue
        is_onset = np.zeros(len(sf.frame_idx), dtype=bool)
        in_run = np.zeros(len(sf.frame_idx), dtype=bool)
        for a, b in bad_runs(sf):
            is_onset[a] = True
            in_run[a:b] = True
        labeled = (
            np.isfinite(sf.err_loma)
            & np.isfinite(sf.err_samh)
            & (np.abs(sf.err_loma - sf.err_samh) > _LABEL_MARGIN)
        )
        y = sf.err_samh < sf.err_loma
        for pop, mask in (("onset", is_onset & labeled), ("sustained", in_run & ~is_onset & labeled)):
            parts[pop]["y"].append(y[mask])
            for f in _FEATURES:
                parts[pop][f].append(sf.feats[f][mask])

    rows: list[list] = []
    for f in _FEATURES:
        cells = []
        for pop in ("onset", "sustained"):
            y = np.concatenate(parts[pop]["y"]) if parts[pop]["y"] else np.zeros(0, bool)
            v = np.concatenate(parts[pop][f]) if parts[pop][f] else np.zeros(0)
            ok = np.isfinite(v)
            a = auc(v[ok], y[ok])
            cells.append(f"{a:.3f}" if np.isfinite(a) else "-")
        n_onset = sum(len(x) for x in parts["onset"]["y"])
        n_sust = sum(len(x) for x in parts["sustained"]["y"])
        rows.append([f, *cells, n_onset, n_sust])
    return rows


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #


def stateless_picks(seqs: list[SeqFrames], tau: float) -> dict[str, np.ndarray]:
    """Baseline switch, plus force samh whenever the branches disagree > tau."""
    return {
        sf.seq: sf.baseline_pick
        | (np.isfinite(sf.feats["disagreement"]) & (sf.feats["disagreement"] > tau))
        for sf in seqs
    }


def latch_picks(
    seqs: list[SeqFrames],
    tau_hi: float,
    tau_lo: float,
    k_exit: int,
    speed_arbiter: bool,
) -> dict[str, np.ndarray]:
    """Two-state selector: enter distrust on a divergence, hold, leave on re-agreement.

    Args:
        seqs: Sequences to run the state machine over (causal, in frame order).
        tau_hi: Disagreement (px) that opens a distrust episode.
        tau_lo: Disagreement (px) below which the branches count as re-agreeing.
        k_exit: Consecutive re-agreeing frames needed to leave the distrust state.
        speed_arbiter: When True, only enter if the LoMA quad is the one that
            jumped (its corner displacement exceeds SAM-H's at the onset frame).
            Divergence alone says the branches disagree, not which one broke.

    Returns:
        seq name -> (N,) bool, True = take the SAM-H candidate.
    """
    out: dict[str, np.ndarray] = {}
    for sf in seqs:
        dis = sf.feats["disagreement"]
        loma_speed, samh_speed = sf.feats["loma_speed"], sf.feats["samh_speed"]
        pick = sf.baseline_pick.copy()
        distrust = False
        held = 0
        for k in range(len(pick)):
            if not distrust:
                diverged = np.isfinite(dis[k]) and dis[k] > tau_hi
                if diverged and speed_arbiter:
                    speeds_known = np.isfinite(loma_speed[k]) and np.isfinite(samh_speed[k])
                    diverged = speeds_known and samh_speed[k] < loma_speed[k]
                if diverged:
                    distrust, held = True, 0
            else:
                held = held + 1 if (np.isfinite(dis[k]) and dis[k] < tau_lo) else 0
                if held >= k_exit:
                    distrust = False
            if distrust:
                pick[k] = True
        out[sf.seq] = pick
    return out


def oracle_episode_picks(seqs: list[SeqFrames], per_episode_vote: bool) -> dict[str, np.ndarray]:
    """Bound on any latch: perfect episode boundaries, one decision per episode.

    Args:
        seqs: Sequences.
        per_episode_vote: When True the episode-wide decision is the majority
            (is SAM-H better on most of the episode's evaluated frames); when
            False every detected episode switches to SAM-H unconditionally.

    Returns:
        seq name -> (N,) bool selection.  This is not deployable — it uses GT to
        find the episodes — it measures how much of the per-frame oracle's
        headroom survives being forced to act at episode granularity.
    """
    out: dict[str, np.ndarray] = {}
    for sf in seqs:
        pick = sf.baseline_pick.copy()
        for a, b in bad_runs(sf):
            take = True
            if per_episode_vote:
                sel = slice(a, b)
                both = np.isfinite(sf.err_samh[sel]) & np.isfinite(sf.err_loma[sel])
                take = bool(both.any() and (sf.err_samh[sel][both] < sf.err_loma[sel][both]).mean() > 0.5)
            pick[a:b] = take
        out[sf.seq] = pick
    return out


def rule_frontier(seqs: list[SeqFrames], train: set[str], test: set[str]) -> list[dict]:
    """Fit every rule family on the train half, score it on the held-out half."""
    results: list[dict] = []

    def add(name: str, picks: dict[str, np.ndarray], p15_train: Optional[float] = None) -> None:
        p_test, switch = evaluate_rule(seqs, picks, test)
        results.append({"rule": name, "p15_train": p15_train, "switch_rate": switch, **p_test})

    add("baseline (the run's inlier gate)", {sf.seq: sf.baseline_pick for sf in seqs})
    add("ORACLE per-frame (ceiling)", oracle_picks(seqs))
    add("ORACLE episode, always samh", oracle_episode_picks(seqs, per_episode_vote=False))
    add("ORACLE episode, majority vote", oracle_episode_picks(seqs, per_episode_vote=True))

    best: Optional[tuple[float, str, dict]] = None
    for tau in _TAU_HI_GRID:
        picks = stateless_picks(seqs, tau)
        p_train, _ = evaluate_rule(seqs, picks, train)
        if best is None or p_train["p15"] > best[0]:
            best = (p_train["p15"], f"stateless: disagreement > {tau:g} -> samh", picks)
    add(best[1], best[2], best[0])

    for arbiter in (False, True):
        best = None
        for tau_hi in _TAU_HI_GRID:
            for frac in _TAU_LO_FRAC_GRID:
                for k_exit in _K_EXIT_GRID:
                    tau_lo = tau_hi * frac
                    picks = latch_picks(seqs, tau_hi, tau_lo, k_exit, arbiter)
                    p_train, _ = evaluate_rule(seqs, picks, train)
                    label = (
                        f"latch{'+speed arbiter' if arbiter else ''}: "
                        f"enter dis>{tau_hi:g}, exit dis<{tau_lo:g} x{k_exit}"
                    )
                    if best is None or p_train["p15"] > best[0]:
                        best = (p_train["p15"], label, picks)
        add(best[1], best[2], best[0])
    return results


# --------------------------------------------------------------------------- #
# failure-type tags
# --------------------------------------------------------------------------- #


def load_tags(path: Path) -> dict[str, list[str]]:
    """Gallery annotations as ``"{seq}_f{frame:05d}" -> [tag, ...]`` (empty if absent)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("ann", {})


def tag_report(
    seqs: list[SeqFrames],
    tags: dict[str, list[str]],
    rules: list[tuple[str, dict[str, np.ndarray]]],
) -> list[list]:
    """Per-tag population stats and per-rule rescue rate on the tagged frames.

    A tag's rescue rate is the fraction of its frames that a rule brings under
    15 px.  Tagged frames are a biased sample (the top of the inlier-desc
    breakdown ranking), so read the columns as per-class behaviour, never as
    class priors.
    """
    if not tags:
        return []
    lookup = {sf.seq: sf for sf in seqs}
    index: dict[str, list[tuple[SeqFrames, int]]] = {}
    for key, tag_list in tags.items():
        seq, _, frame = key.rpartition("_f")
        sf = lookup.get(seq)
        if sf is None:
            continue
        hits = np.nonzero(sf.frame_idx == int(frame))[0]
        if not len(hits):
            continue
        for tag in tag_list:
            index.setdefault(tag, []).append((sf, int(hits[0])))

    rows: list[list] = []
    for tag, entries in sorted(index.items(), key=lambda kv: -len(kv[1])):
        dis = np.asarray([sf.feats["disagreement"][k] for sf, k in entries])
        err = np.asarray([sf.err_loma[k] for sf, k in entries])
        cells = []
        for _, picks in rules:
            fixed = [
                rule_errors(sf, picks[sf.seq])[k] <= _BAD_PX
                for sf, k in entries
                if np.isfinite(rule_errors(sf, picks[sf.seq])[k])
            ]
            cells.append(f"{100 * float(np.mean(fixed)):.0f}%" if fixed else "-")
        rows.append(
            [
                tag,
                len(entries),
                f"{np.nanmedian(err):.0f}",
                f"{np.nanmedian(dis):.0f}",
                *cells,
            ]
        )
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def md_table(header: list[str], rows: list[list]) -> str:
    """Markdown table (also readable as plain text in the HTML ``<pre>``)."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    """Run the whole study and write the report (no arguments by design)."""
    seqs = load_sequences(_PER_SEQ_DIR)
    train, test = train_test_split(seqs)
    n_frames = sum(len(sf.frame_idx) for sf in seqs)
    print(f"substrate : {_PER_SEQ_DIR}")
    print(f"sequences : {len(seqs)} ({len(train)} train / {len(test)} test), frames {n_frames}")

    ep_rows, ep_summary = episode_report(seqs)
    t_episodes = md_table(
        ["population", "frames", "episodes", "median len", "mean len", "max len",
         "% frames in runs >= 5", "% in runs >= 10"],
        ep_rows,
    )

    t_auc = md_table(
        ["feature", "AUC onset", "AUC sustained", "n onset", "n sustained"],
        onset_sustained_aucs(seqs, train),
    )

    results = rule_frontier(seqs, train, test)
    t_rules = md_table(
        ["rule (fit on train, scored on test)", "P@5", "P@15", "P@30", "P@15 train", "changed"],
        [
            [
                r["rule"],
                f"{r['p5']:.2f}",
                f"{r['p15']:.2f}",
                f"{r['p30']:.2f}",
                "-" if r["p15_train"] is None else f"{r['p15_train']:.2f}",
                f"{100 * r['switch_rate']:.1f}%",
            ]
            for r in results
        ],
    )

    tags = load_tags(_TAGS_PATH)
    tag_rules = [
        ("baseline", {sf.seq: sf.baseline_pick for sf in seqs}),
        ("oracle", oracle_picks(seqs)),
    ]
    t_tags = ""
    if tags:
        t_tags = md_table(
            ["tag", "frames", "median err_loma", "median disagreement",
             *[f"rescued: {n}" for n, _ in tag_rules]],
            tag_report(seqs, tags, tag_rules),
        )

    print(f"\n## Episode structure (LoMA error > {_BAD_PX:g} px)\n\n{t_episodes}")
    print(f"\n## Feature AUC: onset vs sustained frames (train split)\n\n{t_auc}")
    print(f"\n## Rule frontier (held-out sequences)\n\n{t_rules}")
    if t_tags:
        print(f"\n## Failure-type tags ({_TAGS_PATH.name}; biased sample — not class priors)"
              f"\n\n{t_tags}")
    else:
        print(f"\n(no tags at {_TAGS_PATH} — copy loma_tags.json there for the tag join)")

    baseline = next(r for r in results if r["rule"].startswith("baseline"))
    oracle = next(r for r in results if r["rule"].startswith("ORACLE per-frame"))
    deployable = [r for r in results if r["rule"][0].islower()]
    best = max(deployable, key=lambda r: r["p15"]) if deployable else None
    summary = {
        "sequences": len(seqs),
        "frames": n_frames,
        **ep_summary,
        "test_baseline": {k: baseline[k] for k in ("p5", "p15", "p30")},
        "test_oracle": {k: oracle[k] for k in ("p5", "p15", "p30")},
        "best_deployable_rule": None if best is None else best["rule"],
        "best_deployable": None if best is None else {k: best[k] for k in ("p5", "p15", "p30")},
    }

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (_OUTPUT_DIR / "rules.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rule", "p5", "p15", "p30", "p15_train", "switch_rate"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>stateful selection (S1 v3)</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1100px}}pre{{font-size:12px}}</style>
</head><body>
<h1>Stateful candidate selection (S1 v3)</h1>
<p>substrate: <code>{_PER_SEQ_DIR}</code></p>
<pre>{json.dumps(summary, indent=2)}</pre>
<h2>Episode structure</h2><pre>{t_episodes}</pre>
<h2>Feature AUC: onset vs sustained</h2><pre>{t_auc}</pre>
<h2>Rule frontier (held-out sequences)</h2><pre>{t_rules}</pre>
{f'<h2>Failure-type tags</h2><pre>{t_tags}</pre>' if t_tags else ''}
</body></html>"""
    (_OUTPUT_DIR / "report.html").write_text(html, encoding="utf-8")
    print(f"\nReport : {_OUTPUT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
