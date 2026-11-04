"""Summarize per-task resource-usage CSVs written by run.sbatch.

Each array task drops a file at  slurm/logs/resources_<jobtag>.csv  containing
rows sampled every interval with these columns:

    timestamp, gpu_mem_used_mib, gpu_mem_total_mib, gpu_util_pct,
    gpu_mem_util_pct, host_mem_used_mib, host_mem_total_mib, load1,
    proc_rss_mib, proc_pcpu_sum, proc_count

Use this script to:
  - confirm that --pack didn't push GPU memory or RAM over the limit
  - pick a higher --pack next time if there's headroom

When --pack and --mail-to are given (run.sbatch passes both), it also emails a
short warning if the pack looks misconfigured — over the GPU-memory limit, or
badly under-utilising the GPU so a larger --pack would help.

Usage:
    python slurm/summarize_resources.py [glob ...]
    python slurm/summarize_resources.py CSV --pack 4 --mail-to me@x.org --job-tag run_123_1

Defaults to slurm/logs/resources_*.csv if no glob is given.
"""

from __future__ import annotations

import argparse
import csv
import glob
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Optional


COLS = [
    "timestamp", "gpu_mem_used_mib", "gpu_mem_total_mib",
    "gpu_util_pct", "gpu_mem_util_pct",
    "host_mem_used_mib", "host_mem_total_mib", "load1",
    "proc_rss_mib", "proc_pcpu_sum", "proc_count",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _read_csv(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {c: [] for c in COLS if c != "timestamp"}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return out
        for row in reader:
            for c in out:
                v = row.get(c, "").strip()
                if not v:
                    continue
                try:
                    out[c].append(float(v))
                except ValueError:
                    pass
    return out


def _summarize_one(path: Path) -> Optional[dict]:
    d = _read_csv(path)
    n = len(d.get("gpu_mem_used_mib", []))
    if n == 0:
        return None

    def stat(col: str) -> tuple[float, float, float]:
        v = d.get(col) or []
        if not v:
            return (float("nan"),) * 3
        return _percentile(v, 0.5), _percentile(v, 0.95), max(v)

    return {
        "path": path, "n": n,
        "gpu_mem": stat("gpu_mem_used_mib"),
        "gpu_util": stat("gpu_util_pct"),
        "rss": stat("proc_rss_mib"),
        "pcpu": stat("proc_pcpu_sum"),
        "gpu_total": max(d.get("gpu_mem_total_mib") or [0.0]),
        "host_total": max(d.get("host_mem_total_mib") or [0.0]),
    }


def _fmt_mib(v: float) -> str:
    return f"{v:7.0f} MiB" if v == v else "      - MiB"


def _fmt_pct(v: float) -> str:
    return f"{v:5.0f}%" if v == v else "    -%"


def suggest_pack(rows: list[dict], headroom: float) -> dict:
    """Compute a suggested --pack and the limiting ceilings.

    Returns dict with worst-case stats and per-resource pack ceilings. Values
    are NaN/None when a resource could not be measured.
    """
    worst_gpu_mem = max(s["gpu_mem"][2] for s in rows)
    med_gpu_p50 = statistics.median(s["gpu_util"][0] for s in rows)
    worst_rss = max(s["rss"][2] for s in rows)
    gpu_total = max(s["gpu_total"] for s in rows) or float("nan")
    host_total = max(s["host_total"] for s in rows) or float("nan")

    mem_pack = util_pack = ram_pack = None
    if worst_gpu_mem > 0 and gpu_total == gpu_total:
        mem_pack = max(1, int((headroom * gpu_total) // worst_gpu_mem))
    if med_gpu_p50 == med_gpu_p50:
        util_pack = max(1, int(100.0 // max(1.0, med_gpu_p50)))
    if host_total == host_total and worst_rss > 0:
        ram_pack = max(1, int((headroom * host_total) // worst_rss))

    ceilings = [p for p in (mem_pack, util_pack, ram_pack) if p is not None]
    suggested = min(ceilings) if ceilings else None

    return {
        "worst_gpu_mem": worst_gpu_mem,
        "med_gpu_p50": med_gpu_p50,
        "worst_rss": worst_rss,
        "gpu_total": gpu_total,
        "host_total": host_total,
        "mem_pack": mem_pack,
        "util_pack": util_pack,
        "ram_pack": ram_pack,
        "suggested": suggested,
    }


def print_summary(rows: list[dict], suggestion: dict, headroom: float) -> None:
    """Print per-file table + overall stats + pack-size suggestion."""
    header = (f"{'file':<48} {'N':>3} "
              f"{'gpu_mem p50':>11} {'p95':>11} {'max':>11} "
              f"{'gpu_util p50':>13} {'max':>5} "
              f"{'rss p95':>10} {'max':>10} "
              f"{'pcpu p95':>10}")
    print(header)
    print("-" * len(header))
    for s in rows:
        gm, gu, rs, pc = s["gpu_mem"], s["gpu_util"], s["rss"], s["pcpu"]
        print(f"{s['path'].name:<48} {s['n']:>3d} "
              f"{_fmt_mib(gm[0])} {_fmt_mib(gm[1])} {_fmt_mib(gm[2])} "
              f"{_fmt_pct(gu[0]):>13} {_fmt_pct(gu[2]):>5} "
              f"{_fmt_mib(rs[1])} {_fmt_mib(rs[2])} "
              f"{_fmt_pct(pc[1]):>10}")

    print()
    print(f"Files summarized:        {len(rows)}")
    print(f"GPU total memory:        {_fmt_mib(suggestion['gpu_total']).strip()}")
    print(f"Host total memory:       {_fmt_mib(suggestion['host_total']).strip()}")
    print(f"Worst-case gpu_mem max:  {_fmt_mib(suggestion['worst_gpu_mem']).strip()}")
    print(f"Worst-case host RSS:     {_fmt_mib(suggestion['worst_rss']).strip()}")
    print(f"GPU util.gpu p50 (median across files): {suggestion['med_gpu_p50']:.0f}%")

    if suggestion["suggested"] is not None:
        print()
        print(f"Suggested --pack: {suggestion['suggested']}")
        if suggestion["mem_pack"] is not None:
            print(f"  - GPU-mem ceiling : {suggestion['mem_pack']:>3d}  "
                  f"({headroom*100:.0f}% of {_fmt_mib(suggestion['gpu_total']).strip()} ÷ "
                  f"{_fmt_mib(suggestion['worst_gpu_mem']).strip()})")
        if suggestion["util_pack"] is not None:
            print(f"  - GPU-util ceiling: {suggestion['util_pack']:>3d}  "
                  f"(100% ÷ median util_p50 {suggestion['med_gpu_p50']:.0f}%)")
        if suggestion["ram_pack"] is not None:
            print(f"  - host-RAM ceiling: {suggestion['ram_pack']:>3d}  "
                  f"({headroom*100:.0f}% of {_fmt_mib(suggestion['host_total']).strip()} ÷ "
                  f"{_fmt_mib(suggestion['worst_rss']).strip()})")


def diagnose_pack(pack: int, suggestion: dict, headroom: float) -> Optional[str]:
    """Return a warning message if the actual --pack looks misconfigured.

    Over-packed: worst-case GPU memory already exceeds the headroom limit, i.e.
    the pack risks OOM. Under-packed: the suggested pack is at least double the
    actual one, i.e. the GPU is sitting idle and a larger pack would help.
    Returns None when the pack looks fine.
    """
    issues: list[str] = []
    gpu_total = suggestion["gpu_total"]
    worst_gpu_mem = suggestion["worst_gpu_mem"]
    suggested = suggestion["suggested"]

    if gpu_total == gpu_total and worst_gpu_mem > headroom * gpu_total:
        issues.append(
            f"OVER-PACKED: worst-case GPU memory {worst_gpu_mem:.0f} MiB exceeds "
            f"{headroom*100:.0f}% of {gpu_total:.0f} MiB — risk of OOM. "
            f"Lower --pack (currently {pack})."
        )
    if suggested is not None and suggested >= 2 * pack and pack >= 1:
        issues.append(
            f"UNDER-PACKED: suggested --pack {suggested} is ≥2× the current {pack}; "
            f"the GPU is under-utilised (util p50 {suggestion['med_gpu_p50']:.0f}%). "
            f"Raise --pack to ~{suggested}."
        )
    return "\n".join(issues) if issues else None


def send_mail(mail_to: str, subject: str, body: str) -> bool:
    """Best-effort email via `mail` then `sendmail`. Returns True on success."""
    if not mail_to:
        return False
    if shutil.which("mail"):
        try:
            subprocess.run(["mail", "-s", subject, mail_to],
                           input=body, text=True, check=True)
            return True
        except Exception:
            pass
    sendmail = shutil.which("sendmail") or "/usr/sbin/sendmail"
    if Path(sendmail).exists():
        try:
            msg = f"Subject: {subject}\nTo: {mail_to}\n\n{body}\n"
            subprocess.run([sendmail, "-t"], input=msg, text=True, check=True)
            return True
        except Exception:
            pass
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("globs", nargs="*", default=["slurm/logs/resources_*.csv"],
                   help="CSV file globs (default: slurm/logs/resources_*.csv).")
    p.add_argument("--headroom", type=float, default=0.85,
                   help="Fraction of a resource to actually use when suggesting a pack size.")
    p.add_argument("--pack", type=int, default=None,
                   help="Actual --pack used; enables misconfiguration diagnosis.")
    p.add_argument("--mail-to", default="",
                   help="Email address to warn when the pack looks misconfigured.")
    p.add_argument("--job-tag", default="",
                   help="Job tag included in the warning email subject.")
    args = p.parse_args()

    paths: list[Path] = []
    for g in args.globs:
        paths.extend(Path(x) for x in sorted(glob.glob(g)))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        print("No CSV files matched.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for path in paths:
        s = _summarize_one(path)
        if s is None:
            print(f"(skip) {path}: no data rows", file=sys.stderr)
            continue
        rows.append(s)

    if not rows:
        print("No usable samples found.", file=sys.stderr)
        sys.exit(1)

    suggestion = suggest_pack(rows, args.headroom)
    print_summary(rows, suggestion, args.headroom)

    # ---- pack misconfiguration diagnosis + optional email -------------------
    if args.pack is not None:
        warning = diagnose_pack(args.pack, suggestion, args.headroom)
        if warning:
            print()
            print("PACK CONFIGURATION WARNING")
            print(warning)
            if args.mail_to:
                subject = f"[flatsam2] pack warning {args.job_tag}".strip()
                body = (
                    f"Pack configuration warning for job {args.job_tag or '(unknown)'}\n\n"
                    f"{warning}\n\n"
                    f"GPU total: {suggestion['gpu_total']:.0f} MiB | "
                    f"worst gpu_mem: {suggestion['worst_gpu_mem']:.0f} MiB | "
                    f"util p50: {suggestion['med_gpu_p50']:.0f}% | "
                    f"suggested --pack: {suggestion['suggested']}\n"
                )
                if send_mail(args.mail_to, subject, body):
                    print(f"(warning emailed to {args.mail_to})")
                else:
                    print(f"(could not send email to {args.mail_to}; warning is in this log)")


if __name__ == "__main__":
    main()
