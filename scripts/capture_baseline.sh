#!/usr/bin/env bash
# Capture the OLD-code golden baseline + noise floor before the C5 extraction
# (see LINEAR_REFACTOR.md). Runs each config TWICE with the same seed, then
# compares the two runs: the max corner diff is the noise floor (GPU non-
# determinism), and any `selected`/`n_inliers`/`corrected` mismatch means the
# determinism is insufficient and must be fixed before refactoring.
#
# Usage:
#   scripts/capture_baseline.sh <S1>[,<S2>,...] [dataset] [out_root]
#
# Example:
#   scripts/capture_baseline.sh V01_1,V02_1 pot /tmp/gold_old
#
# After a refactor commit, capture the NEW code into a fresh dir and compare it
# to run_a, feeding the noise floor printed below as --corner-threshold:
#   python scripts/golden_harness.py capture --config <cfg> --dataset <ds> \
#       --seqs <S1>,<S2> --out /tmp/gold_new
#   python scripts/golden_harness.py compare /tmp/gold_old/<cfg_stem>/run_a \
#       /tmp/gold_new --corner-threshold <floor>

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <seq1[,seq2,...]> [dataset] [out_root]" >&2
    exit 2
fi

SEQS="$1"
DATASET="${2:-pot}"
OUT_ROOT="${3:-/tmp/gold_old}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$REPO_ROOT/scripts/golden_harness.py"

# Configs to baseline: the hairiest path (LoMa + SAM-H + symmetry, decoupled
# sub-tracker) and the flow path that C5 extracts. Add more here if needed.
CONFIGS=(
    "configs/tracker/loma_samh_decoupled.py"
    "configs/tracker/flow_hough_symmetry.py"
)

echo "seqs=$SEQS  dataset=$DATASET  out_root=$OUT_ROOT"
echo "repo=$REPO_ROOT"
echo

for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .py)"
    out_a="$OUT_ROOT/$stem/run_a"
    out_b="$OUT_ROOT/$stem/run_b"

    echo "==================================================================="
    echo ">>> $stem : capturing two same-seed runs"
    echo "==================================================================="
    python "$HARNESS" capture --config "$cfg" --dataset "$DATASET" \
        --seqs "$SEQS" --out "$out_a"
    python "$HARNESS" capture --config "$cfg" --dataset "$DATASET" \
        --seqs "$SEQS" --out "$out_b"

    echo
    echo ">>> $stem : noise floor (run_a vs run_b)"
    # `compare` exits nonzero on any logic (selected/inlier/corrected) mismatch;
    # don't let that abort the loop — we want every config's report.
    python "$HARNESS" compare "$out_a" "$out_b" \
        | tee "$OUT_ROOT/$stem/noise_floor_report.txt" || true
    echo
done

echo "==================================================================="
echo "DONE. Per-config reports under $OUT_ROOT/<stem>/noise_floor_report.txt"
echo "Read 'max corner diff across all seqs' (= noise floor, feed as"
echo "--corner-threshold later) and confirm 'total logic mismatches: 0'."
echo "==================================================================="
