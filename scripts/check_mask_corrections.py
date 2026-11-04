"""Verify mask-correction telemetry exists in tracker results.

Checks that the per-frame ``corrected`` flag (SAM3 mask self-correction) is
actually recorded in the ``{seq}_debug.npz`` sidecars and that at least some
correction events fired — the data the compare_viewer timeline markers rely on.

Scans ``<results-dir>/<config>/<dataset>/`` for debug sidecars, loads the
``corrected`` / ``frame_idx`` arrays, and reports per-config totals plus a few
example (sequence, frame) hits.

Usage:
    python scripts/check_mask_corrections.py                 # all *correct* configs
    python scripts/check_mask_corrections.py --match loma    # configs containing 'loma'
    python scripts/check_mask_corrections.py loma_samh_correct_decoupled_inl35
    python scripts/check_mask_corrections.py --results-dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Allow running as a standalone script from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flatsam2.path_config import paths  # noqa: E402


def _check_config(config_dir: Path, examples: int) -> bool:
    """Report correction stats for one config dir. Returns True if events found."""
    debug_files = sorted(config_dir.rglob("*_debug.npz"))
    if not debug_files:
        print(f"  [no _debug.npz sidecars found under {config_dir}]")
        return False

    n_seqs = 0
    n_seqs_with_corr = 0
    total_corr = 0
    total_frames = 0
    missing_field = 0
    example_hits: List[str] = []

    for f in debug_files:
        seq_name = f.name[: -len("_debug.npz")]
        try:
            data = np.load(f, allow_pickle=False)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: failed to load {f.name}: {e}")
            continue
        n_seqs += 1
        if "corrected" not in data.files or "frame_idx" not in data.files:
            missing_field += 1
            continue
        corrected = data["corrected"].astype(bool)
        frame_idx = data["frame_idx"].astype(int)
        total_frames += len(corrected)
        n = int(corrected.sum())
        if n > 0:
            n_seqs_with_corr += 1
            total_corr += n
            if len(example_hits) < examples:
                hit_frames = frame_idx[: len(corrected)][corrected].tolist()
                preview = hit_frames[:8]
                more = "…" if len(hit_frames) > len(preview) else ""
                example_hits.append(f"    {seq_name}: {n} hits @ frames {preview}{more}")

    # Older runs recorded corrections in a separate {seq}_corrections.json
    # sidecar (before they were unified into the npz). Scan those too — they are
    # the real source of truth when the npz 'corrected' field is stale.
    json_total, json_seqs, json_examples = _scan_corrections_json(
        config_dir, examples)

    rel = config_dir.name
    if missing_field == n_seqs and n_seqs > 0 and json_total == 0:
        print(f"  {rel}: {n_seqs} sidecars but NONE carry 'corrected'/'frame_idx' "
              f"and no _corrections.json (old run? rerun to regenerate)")
        return False

    print(f"  {rel}: {n_seqs} sequences, {total_frames} tracked frames")
    print(f"    npz  'corrected': {total_corr} events across "
          f"{n_seqs_with_corr}/{n_seqs} sequences"
          + (f" ({missing_field} sidecars missing the field)" if missing_field else ""))
    for line in example_hits:
        print(line)
    print(f"    json corrections: {json_total} events across {json_seqs} sequences")
    for line in json_examples:
        print(line)
    return (total_corr > 0) or (json_total > 0)


def _scan_corrections_json(config_dir: Path, examples: int):
    """Sum correction events from legacy {seq}_corrections.json sidecars."""
    files = sorted(config_dir.rglob("*_corrections.json"))
    total = 0
    n_seqs = 0
    example_hits: List[str] = []
    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: failed to read {f.name}: {e}")
            continue
        frames = payload.get("corrected_frames", [])
        if frames:
            total += len(frames)
            n_seqs += 1
            if len(example_hits) < examples:
                rel = f.relative_to(config_dir)
                preview = frames[:8]
                more = "…" if len(frames) > len(preview) else ""
                example_hits.append(
                    f"    {rel}: {len(frames)} hits @ {preview}{more}")
    return total, n_seqs, example_hits


def _load_corners(txt_path: Path) -> np.ndarray:
    """Load a predictions .txt as a (T, 8) array (8 floats per frame)."""
    arr = np.loadtxt(txt_path)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


def _diff_configs(dir_a: Path, dir_b: Path, eps: float, examples: int) -> bool:
    """Diff predictions between two config dirs. Returns True if any differ.

    Compares ``<dataset>/<seq>.txt`` files present in both dirs. This is the
    controlled test for whether correction changed anything: point it at a
    correct config and its exact non-correct twin (only ``correction.enabled``
    differs). Identical predictions => correction never altered the output.
    """
    a_files = {p.relative_to(dir_a): p for p in dir_a.rglob("*.txt")
               if not p.name.endswith("_debug.npz")}
    if not a_files:
        print(f"  no .txt predictions under {dir_a}")
        return False

    n_common = 0
    n_diff_seqs = 0
    total_diff_frames = 0
    max_dev = 0.0
    example_hits: List[str] = []

    for rel, pa in sorted(a_files.items()):
        pb = dir_b / rel
        if not pb.exists():
            continue
        try:
            ca = _load_corners(pa)
            cb = _load_corners(pb)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: failed to load {rel}: {e}")
            continue
        n_common += 1
        t = min(len(ca), len(cb))
        if t == 0:
            continue
        per_frame_dev = np.abs(ca[:t] - cb[:t]).max(axis=1)  # (t,) max coord delta
        diff_frames = int((per_frame_dev > eps).sum())
        if diff_frames > 0:
            n_diff_seqs += 1
            total_diff_frames += diff_frames
            seq_max = float(per_frame_dev.max())
            max_dev = max(max_dev, seq_max)
            if len(example_hits) < examples:
                first = int(np.argmax(per_frame_dev > eps))
                example_hits.append(
                    f"    {rel}: {diff_frames} frames differ, "
                    f"first @ {first}, max Δ {seq_max:.2f}px")

    print(f"  A: {dir_a.name}")
    print(f"  B: {dir_b.name}")
    print(f"  {n_common} sequences compared")
    print(f"  {n_diff_seqs} sequences differ ({total_diff_frames} frames, "
          f"max Δ {max_dev:.2f}px, eps={eps})")
    for line in example_hits:
        print(line)
    return n_diff_seqs > 0


def _find_config_dirs(
    results_dir: Path,
    configs: List[str],
    match: Optional[str],
) -> List[Path]:
    """Resolve config dirs: explicit names, or substring match against subdirs."""
    if configs:
        dirs = []
        for name in configs:
            d = results_dir / name
            if not d.is_dir():
                print(f"WARNING: config dir not found: {d}", file=sys.stderr)
                continue
            dirs.append(d)
        return dirs

    needle = (match or "correct").lower()
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and needle in d.name.lower()
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*",
                        help="Explicit config names (default: substring match via --match)")
    parser.add_argument("--results-dir", type=Path, default=paths.OUTPUT_DIRECTORY,
                        help="Results root holding <config>/<dataset>/ dirs")
    parser.add_argument("--match", default="correct",
                        help="Substring to select config dirs when none are named "
                             "(default: 'correct')")
    parser.add_argument("--examples", type=int, default=3,
                        help="Example sequences with hits to print per config")
    parser.add_argument("--diff", nargs=2, metavar=("CONFIG_A", "CONFIG_B"),
                        help="Instead of scanning telemetry, diff predictions between "
                             "two config dirs (use a correct config and its non-correct "
                             "twin to test whether correction changed anything)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Per-coordinate pixel tolerance for the --diff comparison")
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"Results root not found: {args.results_dir}", file=sys.stderr)
        return 2

    if args.diff:
        dir_a = args.results_dir / args.diff[0]
        dir_b = args.results_dir / args.diff[1]
        for d in (dir_a, dir_b):
            if not d.is_dir():
                print(f"config dir not found: {d}", file=sys.stderr)
                return 2
        print(f"Results root: {args.results_dir}\nDiffing predictions:\n")
        differ = _diff_configs(dir_a, dir_b, args.eps, args.examples)
        print()
        if differ:
            print("Predictions DIFFER => the two runs produced different output.")
            return 0
        print("Predictions IDENTICAL => correction never changed the output.")
        return 1

    config_dirs = _find_config_dirs(args.results_dir, args.configs, args.match)
    if not config_dirs:
        sel = "named" if args.configs else f"matching '{args.match}'"
        print(f"No config dirs {sel} under {args.results_dir}", file=sys.stderr)
        return 2

    print(f"Results root: {args.results_dir}")
    print(f"Checking {len(config_dirs)} config(s):\n")
    any_events = False
    for d in config_dirs:
        if _check_config(d, args.examples):
            any_events = True
        print()

    if any_events:
        print("OK: at least one config recorded mask-correction events.")
        return 0
    print("FAIL: no mask-correction events found in any checked config.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
