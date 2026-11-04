#!/usr/bin/env python3
"""S1 candidate-selector study: how much of the A0 selection ceiling is reachable online?

A0 (``correction_headroom``) put the per-frame candidate-selection ceiling at
P@5 65.6 -> 75.8 on PTi, with 93% of the gain sitting on frames where the run
kept ``loma_H`` (inliers >= 35) while the stored ``samh_H`` was <= 5 px. The
inlier count is uninformative exactly there (median 168 inliers on rescue
frames; raising the switch threshold to 200 buys < 2 pp and costs P@15), so
this study measures which ONLINE signals separate "samh better" from "loma
better" frames, and what P@5/P@15 simple threshold rules achieve on held-out
sequences.

Decision metric: **P@15** (rules are picked by train P@15 and ranked by test
P@15). PlanarTrack GT carries ~5.7 px mean annotation error with ~10% of
annotations beyond 15 px, so P@5 differences sit at the noise floor; P@5 is
reported as a diagnostic column only.

Substrate: the E0+ dry-run run (``loma_samh_correct_decoupled_dryrun``) — it
behaves exactly like the baseline (no corrections applied) but records the
always-on ``corr_*`` telemetry, whose loma-side containment features are
gate-independent online observables. Everything needed lives in
``{seq}_debug.npz`` + GT: no mask IO, login-node friendly.

Online features (all available at frame t before output selection):
    n_inliers       LoMA RANSAC inlier count
    disagreement    mean corner distance (px) between loma_H- and samh_H-warped
                    init corners — the two-branch consistency signal
    spill_loma      (loma_mask & ~sam) / area(loma_mask)      [corr_spill_loma]
    excess_loma     SAM-area fraction deeper than T px outside the loma-induced
                    mask                                       [corr_excess_loma]
    area_log_ratio  |log(area(sam) / area(loma_mask))|   [corr_area_ratio_loma]
    loma_speed      mean corner displacement (px) of the loma_H warp between
                    consecutive frames — per-branch temporal smoothness
    samh_speed      same for samh_H
    speed_gap       loma_speed − samh_speed (positive = loma jumpier)

v1 result (2026-07-11): threshold stumps on the first five features alone do
NOT beat the baseline switch (test P@15 87.92 = baseline; oracle 90.93; best
AUC 0.66). Disagreement is symmetric — it cannot arbitrate WHO broke — hence
the v2 temporal features and the two-sided arbitration family below
("when the branches disagree, trust the temporally smoother one").

Subcommands:
    dump    npz + GT -> per-sequence CSV of candidate errors + features.
    report  CSVs -> feature AUCs, single-feature and paired threshold-rule
            frontier with a sequence-level train/test split, HTML report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from flatsam2.datasets import get_sequences, pot_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error, precision_at_threshold
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.report_utils import get_plotly_script_tag


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

# Derived at report time from the dumped speeds (no re-dump needed): a branch's
# speed excess over its own recent (lagged, causal) median. Camera shake raises
# both branches' absolute speeds — thresholds on raw speed are shake-sensitive;
# the excess over the branch's own moving baseline (and the gap of the two
# excesses) is not.
_DERIVED = ("loma_speed_exc", "samh_speed_exc", "speed_exc_gap")

CSV_HEADER = [
    "frame_idx",
    "evaluated",
    "selected",
    "err_loma",
    "err_samh",
    *_FEATURES,
]

# Frames where one candidate beats the other by less than this margin (px) are
# ambiguous and excluded from the AUC labels (not from rule evaluation). 3 px
# keeps the labels clear of the PT annotation-noise floor.
_LABEL_MARGIN = 3.0


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #


def _load_gt_fn(dataset: str, initfix_dir: Path) -> Callable:
    """Return the ``load_gt(base_dir, seq)`` callable for 'pot' / 'pti'."""
    if dataset == "pot":
        return pot_load_gt
    if dataset == "pti":
        return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)
    raise ValueError(f"unsupported dataset {dataset!r} (use 'pot' or 'pti')")


def _candidate_H(data: np.lib.npyio.NpzFile, key: str, k: int) -> Optional[np.ndarray]:
    """The k-th stored (3, 3) candidate homography, or None when NaN/absent."""
    if key not in data:
        return None
    H = data[key][k]
    return H if np.isfinite(H).all() else None


def _dump_sequence(
    seq: str,
    load_gt: Callable,
    base_dir: Path,
    run_dir: Path,
    out_csv: Path,
) -> int:
    """Join one sequence's dry-run npz with GT into a per-frame feature CSV.

    Returns:
        Number of evaluated frames written.

    Raises:
        FileNotFoundError: When the debug sidecar is missing.
        ValueError: When the sidecar lacks the ``corr_*`` telemetry (wrong
            substrate run — use a correction-enabled config, e.g. the dry run).
    """
    npz_path = run_dir / f"{seq}_debug.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")
    data = np.load(npz_path)
    for required in ("frame_idx", "loma_H", "samh_H", "corr_spill_loma"):
        if required not in data:
            raise ValueError(
                f"{npz_path} lacks '{required}' — dump needs a correction-enabled "
                "run (E0+ dry run), not the plain baseline"
            )

    gt_corners, valid = load_gt(base_dir, seq)
    init_corners = gt_corners[0]
    frame_ids = data["frame_idx"]

    rows: list[list] = []
    n_eval = 0
    prev_i: Optional[int] = None
    prev_corners: dict[str, Optional[np.ndarray]] = {"loma": None, "samh": None}
    for k in range(len(frame_ids)):
        i = int(frame_ids[k])
        evaluated = bool(valid[i]) if i < len(valid) else False
        loma_H = _candidate_H(data, "loma_H", k)
        samh_H = _candidate_H(data, "samh_H", k)
        loma_c = warp_corners(loma_H, init_corners) if loma_H is not None else None
        samh_c = warp_corners(samh_H, init_corners) if samh_H is not None else None

        err_loma = err_samh = float("nan")
        if evaluated and i < len(gt_corners):
            if loma_c is not None:
                err_loma = float(alignment_error(gt_corners[i], loma_c))
            if samh_c is not None:
                err_samh = float(alignment_error(gt_corners[i], samh_c))
            n_eval += 1

        disagreement = float("nan")
        if loma_c is not None and samh_c is not None:
            disagreement = float(np.linalg.norm(loma_c - samh_c, axis=0).mean())

        # Per-branch temporal smoothness: corner displacement since the
        # previous stored frame (NaN across gaps and at the first frame).
        contiguous = prev_i is not None and i == prev_i + 1

        def _speed(cur: Optional[np.ndarray], prev: Optional[np.ndarray]) -> float:
            if not contiguous or cur is None or prev is None:
                return float("nan")
            return float(np.linalg.norm(cur - prev, axis=0).mean())

        loma_speed = _speed(loma_c, prev_corners["loma"])
        samh_speed = _speed(samh_c, prev_corners["samh"])
        speed_gap = loma_speed - samh_speed
        prev_i = i
        prev_corners = {"loma": loma_c, "samh": samh_c}

        area_ratio = (
            float(data["corr_area_ratio_loma"][k])
            if "corr_area_ratio_loma" in data
            else float("nan")
        )
        area_log_ratio = (
            float(abs(np.log(area_ratio)))
            if np.isfinite(area_ratio) and area_ratio > 0
            else float("nan")
        )

        rows.append(
            [
                i,
                int(evaluated),
                str(data["selected"][k]) if "selected" in data else "",
                err_loma,
                err_samh,
                int(data["n_inliers"][k]) if "n_inliers" in data else -1,
                disagreement,
                float(data["corr_spill_loma"][k]),
                float(data["corr_excess_loma"][k]) if "corr_excess_loma" in data else float("nan"),
                area_log_ratio,
                loma_speed,
                samh_speed,
                speed_gap,
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([f"{x:.6g}" if isinstance(x, float) else x for x in r])
    tmp.replace(out_csv)
    return n_eval


def cmd_dump(args: argparse.Namespace) -> None:
    """Run the per-frame join over a sequence list."""
    load_gt = _load_gt_fn(args.dataset, args.initfix_dir)
    sequences = get_sequences(
        args.dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split if args.dataset != "pot" else None,
    )
    out_dir = args.output_dir / "per_sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset : {args.dataset} ({DATASETS[args.dataset].label})")
    print(f"Run dir : {args.run_dir}")
    print(f"Output  : {out_dir}")

    total_eval = 0
    failed: list[str] = []
    for k, seq in enumerate(sequences, 1):
        out_csv = out_dir / f"{seq}.csv"
        if out_csv.exists() and args.skip_completed:
            print(f"[{k}/{len(sequences)}] skip {seq} (exists)")
            continue
        try:
            n = _dump_sequence(seq, load_gt, args.base_dir, args.run_dir, out_csv)
            total_eval += n
            print(f"[{k}/{len(sequences)}] {seq}: {n} eval frames")
        except Exception as e:  # noqa: BLE001 - keep going, record failure
            failed.append(seq)
            print(f"[{k}/{len(sequences)}] {seq}: FAILED ({e})")
    print(f"\nDone. eval frames={total_eval} failed={len(failed)}")
    if failed:
        print("failed:", " ".join(failed))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def _lagged_median(v: np.ndarray, window: int = 15, min_count: int = 5) -> np.ndarray:
    """Causal rolling median over the previous ``window`` values (NaN-skipping).

    Strictly lagged (excludes the current value), so a breakdown onset does not
    immediately contaminate its own baseline.
    """
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - window) : i]
        w = w[np.isfinite(w)]
        if len(w) >= min_count:
            out[i] = np.median(w)
    return out


def _load_frames(per_seq_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate evaluated frames from all per-sequence CSVs into columns.

    Derived speed-excess features are computed per sequence on the full frame
    series (including non-evaluated frames) before the evaluated filter, so
    the rolling baselines see contiguous history.
    """
    numeric = [c for c in CSV_HEADER if c != "selected"]
    files = sorted(per_seq_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-sequence CSVs in {per_seq_dir}")
    cols: dict[str, list] = {c: [] for c in [*numeric, *_DERIVED]}
    selected_col: list[str] = []
    seq_col: list[str] = []
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
        if not raw:
            continue
        if "loma_speed" not in raw[0]:
            raise ValueError(
                f"{path} lacks the v2 speed columns — re-run "
                "`selector_study.py dump --no-skip-completed`"
            )
        per = {c: np.asarray([float(d[c]) for d in raw]) for c in numeric}
        per["loma_speed_exc"] = per["loma_speed"] - _lagged_median(per["loma_speed"])
        per["samh_speed_exc"] = per["samh_speed"] - _lagged_median(per["samh_speed"])
        per["speed_exc_gap"] = per["loma_speed_exc"] - per["samh_speed_exc"]
        keep = per["evaluated"] == 1
        for c in [*numeric, *_DERIVED]:
            cols[c].extend(per[c][keep].tolist())
        selected_col.extend(d["selected"] for d, k in zip(raw, keep) if k)
        seq_col.extend([path.stem] * int(keep.sum()))
    out = {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}
    out["selected"] = np.asarray(selected_col, dtype=object)
    out["seq"] = np.asarray(seq_col, dtype=object)
    return out


def _rule_errors(
    err_loma: np.ndarray,
    err_samh: np.ndarray,
    pick_samh: np.ndarray,
) -> np.ndarray:
    """Per-frame error of a selection rule, falling back to the available candidate."""
    err = np.where(pick_samh, err_samh, err_loma)
    err = np.where(np.isfinite(err), err, np.where(pick_samh, err_loma, err_samh))
    return err


def _p5_p15(err: np.ndarray) -> Tuple[float, float]:
    """(P@5, P@15) over the finite entries of ``err``."""
    finite = err[np.isfinite(err)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    return (
        float(precision_at_threshold(finite, 5.0)),
        float(precision_at_threshold(finite, 15.0)),
    )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC of ``scores`` for binary ``labels`` (ties get midranks)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    # midranks for ties
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


def cmd_report(args: argparse.Namespace) -> None:
    """CSVs -> AUC table + threshold-rule frontier + HTML report."""
    data = _load_frames(args.per_seq_dir)
    err_loma, err_samh = data["err_loma"], data["err_samh"]
    seqs = data["seq"]

    # Sequence-level split: even index train, odd index test (deterministic).
    uniq = sorted(set(seqs))
    train_seqs = set(uniq[0::2])
    in_train = np.array([s in train_seqs for s in seqs])
    print(f"frames={len(seqs)}  sequences={len(uniq)} (train {len(train_seqs)})")

    # Baseline rule = what the run did; oracle = per-frame best candidate.
    baseline_pick = data["selected"] == "samh_H"
    oracle_err = np.fmin(
        np.where(np.isfinite(err_loma), err_loma, np.inf),
        np.where(np.isfinite(err_samh), err_samh, np.inf),
    )
    oracle_err = np.where(np.isfinite(oracle_err), oracle_err, np.nan)

    # AUC labels: samh clearly better vs loma clearly better (both available).
    both = np.isfinite(err_loma) & np.isfinite(err_samh)
    labeled = both & (np.abs(err_loma - err_samh) > _LABEL_MARGIN)
    y = err_samh < err_loma  # True = switch to samh was right

    rows_auc: list[list] = []
    oriented: dict[str, np.ndarray] = {}
    auc_by_feat: dict[str, float] = {}
    for feat in (*_FEATURES, *_DERIVED):
        v = data[feat].copy()
        ok = labeled & np.isfinite(v)
        auc = _auc(v[ok & in_train], y[ok & in_train])
        flip = np.isfinite(auc) and auc < 0.5
        oriented[feat] = -v if flip else v
        auc_by_feat[feat] = (1 - auc if flip else auc) if np.isfinite(auc) else 0.5
        rows_auc.append(
            [
                feat + (" (flipped)" if flip else ""),
                f"{auc_by_feat[feat]:.3f}" if np.isfinite(auc) else "-",
                f"{np.isfinite(v).mean() * 100:.0f}%",
            ]
        )

    def evaluate(pick: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        return _p5_p15(_rule_errors(err_loma, err_samh, pick)[mask])

    # Rule frontier: baseline pick OR (oriented feature > tau); tau over train
    # quantiles. Singles, then pairs (AND of the two best singles' grids), then
    # the two-sided arbitration family. Rules identical to the baseline pick
    # (e.g. a threshold below the existing inlier gate) are dropped as no-ops.
    quants = np.linspace(0.05, 0.99, args.n_thresholds)
    results: list[dict] = []

    def add_result(rule: str, feats: list[str], p15_train: float, pick: np.ndarray) -> None:
        changed = pick != baseline_pick
        if not changed.any():
            return
        p5_test, p15_test = evaluate(pick, ~in_train)
        results.append(
            {
                "rule": rule,
                "features": feats,
                "p15_train": p15_train,
                "p15_test": p15_test,
                "p5_test": p5_test,
                "switch_rate": float(changed.mean()),
            }
        )

    for feat in (*_FEATURES, *_DERIVED):
        v = oriented[feat]
        ok_train = in_train & np.isfinite(v)
        taus = np.quantile(v[ok_train], quants) if ok_train.any() else []
        best = None
        for tau in np.unique(taus):
            pick = baseline_pick | (np.isfinite(v) & (v > tau))
            _, p15_train = evaluate(pick, in_train)
            if best is None or p15_train > best[1]:
                best = (float(tau), p15_train, pick)
        if best is None:
            continue
        tau, p15_train, pick = best
        add_result(f"{feat} > {tau:.3g}", [feat], p15_train, pick)

    pair_feats = sorted(auc_by_feat, key=lambda f: -auc_by_feat[f])[:4]
    for a_i in range(len(pair_feats)):
        for b_i in range(a_i + 1, len(pair_feats)):
            fa, fb = pair_feats[a_i], pair_feats[b_i]
            va, vb = oriented[fa], oriented[fb]
            ok_train = in_train & np.isfinite(va) & np.isfinite(vb)
            if not ok_train.any():
                continue
            taus_a = np.unique(np.quantile(va[ok_train], quants))
            taus_b = np.unique(np.quantile(vb[ok_train], quants))
            best = None
            for ta in taus_a:
                for tb in taus_b:
                    pick = baseline_pick | (
                        np.isfinite(va) & (va > ta) & np.isfinite(vb) & (vb > tb)
                    )
                    _, p15_train = evaluate(pick, in_train)
                    if best is None or p15_train > best[2]:
                        best = (float(ta), float(tb), p15_train, pick)
            ta, tb, p15_train, pick = best
            add_result(f"{fa} > {ta:.3g} AND {fb} > {tb:.3g}", [fa, fb], p15_train, pick)

    # Two-sided arbitration: when the branches disagree by more than tau px,
    # ignore the inlier gate and trust the temporally smoother branch. This is
    # the only family that can also move samh-selected frames back to loma.
    # "raw" compares absolute speeds (shake-sensitive); "exc" compares each
    # branch's excess over its own recent baseline (shake-robust).
    dis = data["disagreement"]
    for tag, la, sa in (
        ("raw", "loma_speed", "samh_speed"),
        ("exc", "loma_speed_exc", "samh_speed_exc"),
    ):
        ls, ss = data[la], data[sa]
        ok_arb = np.isfinite(dis) & np.isfinite(ls) & np.isfinite(ss)
        ok_train_arb = in_train & ok_arb
        if not ok_train_arb.any():
            continue
        best = None
        for tau in np.unique(np.quantile(dis[ok_train_arb], quants)):
            pick = baseline_pick.copy()
            m = ok_arb & (dis > tau)
            pick[m] = ss[m] < ls[m]
            _, p15_train = evaluate(pick, in_train)
            if best is None or p15_train > best[1]:
                best = (float(tau), p15_train, pick)
        tau, p15_train, pick = best
        add_result(
            f"ARB({tag}): disagreement > {tau:.3g} -> pick smoother branch",
            ["disagreement", la, sa],
            p15_train,
            pick,
        )

    results.sort(key=lambda r: -r["p15_test"])
    base_test = evaluate(baseline_pick, ~in_train)
    oracle_test = _p5_p15(oracle_err[~in_train])

    # Direction split of the test-split oracle headroom: which side of the
    # switch holds the rescuable P@15 misses.
    test = ~in_train
    base_err = _rule_errors(err_loma, err_samh, baseline_pick)
    miss_test = test & np.isfinite(base_err) & (base_err > 15.0)
    to_samh = int((miss_test & ~baseline_pick & np.isfinite(err_samh) & (err_samh <= 15.0)).sum())
    to_loma = int((miss_test & baseline_pick & np.isfinite(err_loma) & (err_loma <= 15.0)).sum())
    print(
        f"test-split P@15 misses: {int(miss_test.sum())}  rescuable by switching "
        f"loma->samh: {to_samh}  samh->loma: {to_loma}"
    )

    header = ["rule", "p@15 train", "p@15 test", "p@5 test", "changed rate"]
    table_rows = [
        ["baseline (run's switch)", "-", f"{base_test[1]:.2f}", f"{base_test[0]:.2f}", "0%"],
        ["oracle (min candidate)", "-", f"{oracle_test[1]:.2f}", f"{oracle_test[0]:.2f}", "-"],
    ] + [
        [
            r["rule"],
            f"{r['p15_train']:.2f}",
            f"{r['p15_test']:.2f}",
            f"{r['p5_test']:.2f}",
            f"{r['switch_rate'] * 100:.1f}%",
        ]
        for r in results
    ]

    def md_table(hdr: list[str], rws: list[list]) -> str:
        lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
        lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rws]
        return "\n".join(lines)

    t_auc = md_table(["feature (oriented: high = switch)", "train AUC", "available"], rows_auc)
    t_rules = md_table(header, table_rows)
    print(f"\n## Feature AUCs (label: samh better by > {_LABEL_MARGIN} px)\n\n{t_auc}")
    print(f"\n## Rule frontier (test split)\n\n{t_rules}")

    summary = {
        "frames": int(len(seqs)),
        "test_p15_baseline": base_test[1],
        "test_p5_baseline": base_test[0],
        "test_p15_oracle": oracle_test[1],
        "test_p5_oracle": oracle_test[0],
        "best_rule": results[0]["rule"] if results else None,
        "best_rule_p15_test": results[0]["p15_test"] if results else None,
        "best_rule_p5_test": results[0]["p5_test"] if results else None,
        "test_misses_p15": int(miss_test.sum()),
        "test_rescuable_to_samh": to_samh,
        "test_rescuable_to_loma": to_loma,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "rules.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["rule", "features", "p15_train", "p15_test", "p5_test", "switch_rate"]
        )
        w.writeheader()
        for r in results:
            w.writerow({**r, "features": "+".join(r["features"])})

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>selector study</title>
{get_plotly_script_tag()}
<style>body{{font-family:sans-serif;margin:2em}}pre{{font-size:12px}}</style>
</head><body>
<h1>Candidate-selector study (S1)</h1>
<p>substrate: <code>{args.per_seq_dir}</code></p>
<pre>{json.dumps(summary, indent=2)}</pre>
<h2>Feature AUCs</h2><pre>{t_auc}</pre>
<h2>Rule frontier (sequence-level train/test split)</h2><pre>{t_rules}</pre>
</body></html>"""
    (args.output_dir / "report.html").write_text(html, encoding="utf-8")
    print(f"\nSummary: {json.dumps(summary)}")
    print(f"Report : {args.output_dir / 'report.html'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the dump/report CLI."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["pti", "pot"], default="pti")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="dry-run npz + GT -> per-sequence feature CSV")
    d.add_argument("--run-dir", type=Path, default=None,
                   help="correction-enabled run dir with corr_* telemetry "
                        "(default: <OUTPUT>/loma_samh_correct_decoupled_dryrun/<label>)")
    d.add_argument("--base-dir", type=Path, default=None)
    d.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    d.add_argument("--split", choices=["train", "test"], default="test")
    d.add_argument("--sequences", nargs="+")
    d.add_argument("--subset")
    d.add_argument("--subset-file", type=Path)
    d.add_argument("--output-dir", type=Path, default=None)
    d.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    d.set_defaults(func=cmd_dump)

    r = sub.add_parser("report", help="per-sequence CSVs -> AUCs + rule frontier + HTML")
    r.add_argument("--per-seq-dir", type=Path, default=None)
    r.add_argument("--output-dir", type=Path, default=None)
    r.add_argument("--n-thresholds", type=int, default=20,
                   help="quantile grid size per feature")
    r.set_defaults(func=cmd_report)
    return p


def _fill_defaults(args: argparse.Namespace) -> None:
    """Resolve dataset-dependent path defaults left as None by the parser."""
    label = DATASETS[args.dataset].label
    out_default = paths.OUTPUT_DIRECTORY / "experiments" / "selector_study" / label
    if getattr(args, "output_dir", None) is None:
        args.output_dir = out_default
    if getattr(args, "per_seq_dir", None) is None and args.cmd == "report":
        args.per_seq_dir = out_default / "per_sequence"
    if args.cmd == "dump":
        if args.run_dir is None:
            args.run_dir = (
                paths.OUTPUT_DIRECTORY / "loma_samh_correct_decoupled_dryrun" / label
            )
        if args.base_dir is None:
            args.base_dir = paths.POT_BASE if args.dataset == "pot" else paths.PT_BASE


def main() -> None:
    args = build_parser().parse_args()
    _fill_defaults(args)
    args.func(args)


if __name__ == "__main__":
    main()
