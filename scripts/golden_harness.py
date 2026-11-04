#!/usr/bin/env python3
"""Golden-output verification harness for the linear refactor (see LINEAR_REFACTOR.md).

The refactor is pure code-motion: identical calls in identical order must yield
identical output. SAM3 / ptlflow / DINO are GPU nets, so two runs of the *same*
code differ by a small noise floor. This harness separates that noise from real
logic changes:

  * corner deltas  -> float noise (tolerated up to the measured floor)
  * `selected` / `n_inliers` / `corrected` mismatches -> LOGIC change (must be 0)

Workflow (run where the package is installed; not on the dev box):

  # 1. capture two same-seed runs of OLD code -> noise floor
  python scripts/golden_harness.py capture --config configs/tracker/loma_samh_decoupled.py \
      --dataset pot --seqs S1,S2 --out /tmp/gold_old_a
  python scripts/golden_harness.py capture ... --out /tmp/gold_old_b
  python scripts/golden_harness.py compare /tmp/gold_old_a /tmp/gold_old_b   # floor

  # 2. after each refactor commit, capture NEW code and compare to OLD
  python scripts/golden_harness.py capture ... --out /tmp/gold_new
  python scripts/golden_harness.py compare /tmp/gold_old_a /tmp/gold_new

`compare` exits nonzero if any selection/inlier/corrected value differs (logic
change), regardless of float noise.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _leaf_dir(out_dir: Path, config: Path, dataset: str) -> Path:
    """Resolve the leaf dir run.py writes into: out/<config_stem>/<dataset_label>."""
    # Imported lazily so `compare` works without the full package installed.
    from flatsam2.config import compute_output_dir

    return compute_output_dir(out_dir, dataset, config)


def capture(args: argparse.Namespace) -> int:
    """Run run.py (deterministic) on a fixed sequence list, then hash outputs."""
    seqs = [s for s in args.seqs.split(",") if s]
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "run.py"),
        args.dataset,
        "--config", str(args.config),
        "--output-dir", str(args.out),
        "--sequences", *seqs,
        "--deterministic", "--seed", str(args.seed),
        "--no-skip-completed",
    ]
    print("running:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=_REPO_ROOT).returncode
    if rc != 0:
        print(f"run.py exited {rc}", file=sys.stderr)
        return rc

    leaf = _leaf_dir(args.out, args.config, args.dataset)
    digest = _hash_dir(leaf, seqs)
    (leaf / "golden_hash.txt").write_text(digest + "\n")
    print(f"captured -> {leaf}\nsha256={digest}")
    return 0


def _seq_files(leaf: Path, seqs: list[str]) -> list[str]:
    """Sequence names that have a corner .txt under *leaf* (filtered by *seqs* if given)."""
    names = sorted(p.stem for p in leaf.glob("*.txt") if p.stem != "golden_hash")
    if seqs:
        names = [n for n in names if n in seqs]
    return names


def _resolve_leaf(d: Path) -> Path:
    """Accept either the leaf result dir or a parent of it.

    `capture --out` takes a TOP-LEVEL dir; run.py then writes into
    ``<out>/<config_stem>/<dataset_label>/``. So `compare` may be handed either the
    leaf (with corner .txt files) or the top-level. Descend to the unique result dir.
    """
    if _seq_files(d, []):
        return d
    candidates = sorted({p.parent for p in d.rglob("*.txt")})
    candidates = [c for c in candidates if _seq_files(c, [])]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return d  # caller reports "no corner .txt files"
    raise SystemExit(
        f"ambiguous: multiple result dirs under {d}; pass one explicitly:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def _hash_dir(leaf: Path, seqs: list[str]) -> str:
    """Stable sha256 over corner text + debug arrays for the given sequences."""
    h = hashlib.sha256()
    for name in _seq_files(leaf, seqs):
        h.update(name.encode())
        h.update((leaf / f"{name}.txt").read_bytes())
        npz = leaf / f"{name}_debug.npz"
        if npz.exists():
            with np.load(npz, allow_pickle=False) as data:
                for key in sorted(data.files):
                    h.update(key.encode())
                    h.update(np.ascontiguousarray(data[key]).tobytes())
    return h.hexdigest()


def _compare_seq(a: Path, b: Path, name: str) -> dict:
    """Compare one sequence's corners + debug arrays between two capture dirs."""
    # atleast_2d: np.loadtxt collapses a single-frame file to 1-D.
    ca = np.atleast_2d(np.loadtxt(a / f"{name}.txt"))
    cb = np.atleast_2d(np.loadtxt(b / f"{name}.txt"))
    res: dict = {"name": name}
    if ca.shape != cb.shape:
        res["error"] = f"corner shape {ca.shape} vs {cb.shape}"
        return res
    res["max_corner_diff"] = float(np.max(np.abs(ca - cb))) if ca.size else 0.0

    npz_a, npz_b = a / f"{name}_debug.npz", b / f"{name}_debug.npz"
    if npz_a.exists() and npz_b.exists():
        with np.load(npz_a, allow_pickle=False) as da, np.load(npz_b, allow_pickle=False) as db:
            res["selected_mismatch"] = int(np.sum(da["selected"] != db["selected"]))
            res["n_inliers_mismatch"] = int(np.sum(da["n_inliers"] != db["n_inliers"]))
            res["corrected_mismatch"] = int(np.sum(da["corrected"] != db["corrected"]))
    else:
        # A missing debug npz means logic checks could not run — treat as failure,
        # not a silent pass (e.g. new code crashed before writing it).
        res["error"] = "missing debug npz"
    return res


def compare(args: argparse.Namespace) -> int:
    """Diff two capture dirs; nonzero exit on any logic (selection) mismatch."""
    a, b = _resolve_leaf(Path(args.dir_a)), _resolve_leaf(Path(args.dir_b))
    seqs = [s for s in args.seqs.split(",") if s] if args.seqs else []
    names = _seq_files(a, seqs)
    if not names:
        print(f"no corner .txt files found under {a}", file=sys.stderr)
        return 2

    logic_mismatch = 0
    max_corner = 0.0
    print(f"comparing\n  A={a}\n  B={b}")
    print(f"{'sequence':<28} {'max_corner_diff':>16} {'sel':>5} {'inl':>5} {'corr':>5}")
    for name in names:
        r = _compare_seq(a, b, name)
        if "error" in r:
            print(f"{name:<28} ERROR: {r['error']}")
            logic_mismatch += 1
            continue
        sel = r.get("selected_mismatch", 0)
        inl = r.get("n_inliers_mismatch", 0)
        corr = r.get("corrected_mismatch", 0)
        logic_mismatch += sel + inl + corr
        max_corner = max(max_corner, r["max_corner_diff"])
        print(f"{name:<28} {r['max_corner_diff']:>16.6e} {sel:>5} {inl:>5} {corr:>5}")

    print("-" * 64)
    print(f"max corner diff across all seqs: {max_corner:.6e}")
    print(f"total logic mismatches (selected+n_inliers+corrected): {logic_mismatch}")

    # Corner deltas alone do not fail the run UNLESS a noise-floor threshold is given:
    # a homography math bug can change corners while leaving selection labels intact.
    # Feed --corner-threshold the floor measured from two same-seed OLD-code runs.
    corner_fail = args.corner_threshold is not None and max_corner > args.corner_threshold
    if args.corner_threshold is not None:
        print(f"corner threshold (noise floor): {args.corner_threshold:.6e}"
              f" -> {'EXCEEDED' if corner_fail else 'within floor'}")
    else:
        print("corner threshold: none given — corner delta is informational only; "
              "compare it to your measured floor by eye.")

    if logic_mismatch or corner_fail:
        reasons = []
        if logic_mismatch:
            reasons.append("selection/inliers/corrected differ")
        if corner_fail:
            reasons.append("corner diff exceeds noise floor")
        print(f"VERDICT: FAIL — {'; '.join(reasons)}.")
        return 1
    print("VERDICT: PASS — logic identical; corner diff within tolerance.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capture", help="run tracker deterministically and hash outputs")
    pc.add_argument("--config", type=Path, required=True)
    pc.add_argument("--dataset", default="pot")
    pc.add_argument("--seqs", required=True, help="comma-separated sequence names")
    pc.add_argument("--out", type=Path, required=True, help="top-level output dir")
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=capture)

    pp = sub.add_parser("compare", help="diff two capture leaf dirs")
    pp.add_argument("dir_a", help="capture dir A (leaf or top-level --out; auto-descends)")
    pp.add_argument("dir_b", help="capture dir B")
    pp.add_argument("--seqs", help="restrict to these comma-separated sequences")
    pp.add_argument(
        "--corner-threshold",
        type=float,
        default=None,
        help=(
            "Noise floor in pixels (max corner diff from two same-seed OLD-code runs). "
            "When set, FAIL if new-vs-old max corner diff exceeds it — catches "
            "homography math changes that don't alter selection labels."
        ),
    )
    pp.set_defaults(func=compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
