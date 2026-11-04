#!/usr/bin/env bash
# Verify the CURRENT (refactored) code is byte-identical to the OLD-code golden
# baseline captured by capture_baseline.sh (see LINEAR_REFACTOR.md). For each
# config: capture the current checkout, then compare it to the baseline's run_a,
# feeding the measured noise floor (read from the baseline's
# noise_floor_report.txt) as --corner-threshold.
#
# PASS = zero `selected`/`n_inliers`/`corrected` mismatches AND max corner diff
# within the noise floor. A logic bug shows as a selection flip or a corner diff
# above the floor.
#
# Usage:
#   scripts/verify_refactor.sh <seq1[,seq2,...]> [dataset] [baseline_root] [new_root]
#
# Example (mirrors the capture invocation, same seqs/dataset/baseline dir):
#   scripts/verify_refactor.sh Seq_00775,Seq_00003,Seq_00341,Seq_00926 pti \
#       /mnt/personal/serycjon/flatsam2/gold/old /mnt/personal/serycjon/flatsam2/gold/new
#
# Run it on the linear-refactor branch (the code under test is the checkout).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <seq1[,seq2,...]> [dataset] [baseline_root] [new_root]" >&2
    exit 2
fi

SEQS="$1"
DATASET="${2:-pot}"
BASELINE_ROOT="${3:-/tmp/gold_old}"
NEW_ROOT="${4:-/tmp/gold_new}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$REPO_ROOT/scripts/golden_harness.py"

# Must match the CONFIGS list in capture_baseline.sh so the baseline dirs line up.
CONFIGS=(
    "configs/tracker/loma_samh_decoupled.py"
    "configs/tracker/flow_hough_symmetry.py"
)

echo "seqs=$SEQS  dataset=$DATASET"
echo "baseline=$BASELINE_ROOT  new=$NEW_ROOT"
echo "branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)  commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
echo

fail=0
for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .py)"
    base_a="$BASELINE_ROOT/$stem/run_a"
    new_out="$NEW_ROOT/$stem/run_new"
    floor_report="$BASELINE_ROOT/$stem/noise_floor_report.txt"

    if [[ ! -d "$base_a" ]]; then
        echo ">>> $stem : SKIP — no OLD baseline at $base_a" >&2
        continue
    fi

    # Read the measured noise floor from the baseline report (default 0).
    floor="0"
    if [[ -f "$floor_report" ]]; then
        floor="$(grep -oE 'max corner diff across all seqs:[[:space:]]*[0-9.eE+-]+' \
            "$floor_report" | grep -oE '[0-9.eE+-]+$' | tail -n1 || echo 0)"
        [[ -z "$floor" ]] && floor="0"
    fi

    echo "==================================================================="
    echo ">>> $stem : capturing CURRENT code"
    echo "==================================================================="
    python "$HARNESS" capture --config "$cfg" --dataset "$DATASET" \
        --seqs "$SEQS" --out "$new_out"

    echo
    echo ">>> $stem : compare baseline run_a vs NEW (--corner-threshold $floor)"
    mkdir -p "$NEW_ROOT/$stem"
    # `compare` exits nonzero on any logic mismatch or corner diff above the
    # threshold; capture that without aborting the loop so every config reports.
    if python "$HARNESS" compare "$base_a" "$new_out" --corner-threshold "$floor" \
        | tee "$NEW_ROOT/$stem/verify_report.txt"; then
        echo ">>> $stem : PASS"
    else
        echo ">>> $stem : FAIL — see $NEW_ROOT/$stem/verify_report.txt" >&2
        fail=1
    fi
    echo
done

echo "==================================================================="
if [[ "$fail" -eq 0 ]]; then
    echo "ALL CONFIGS PASS — refactor byte-identical to baseline (within floor)."
else
    echo "SOME CONFIGS FAILED — investigate before continuing the refactor."
fi
echo "Reports under $NEW_ROOT/<stem>/verify_report.txt"
echo "==================================================================="
exit "$fail"
