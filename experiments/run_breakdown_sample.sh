#!/usr/bin/env bash
# Unbiased breakdown-tagging round: refresh the ranked list, draw a
# disagreement-stratified random sample from it, and submit the GPU render of
# that sample as an annotatable gallery.
#
# The first tagging round used the inlier-descending ranking, so its class
# counts are not class shares (high-inlier frames are over-represented, and the
# top of that ranking is enriched for frames where LoMA is fine and the GT is
# wrong).  This draw has a known per-stratum sampling rate, so its tag counts
# scale back to the full breakdown population.
#
# Usage (cluster, from anywhere — all paths are absolute):
#   experiments/run_breakdown_sample.sh          # list + sample + submit render
#   experiments/run_breakdown_sample.sh status   # what exists so far
#
# Override with env vars: N_FRAMES (150), MAX_PER_SEQ (4), SEED (0),
# OUTPUT_DIR (path_config output root), TIME (08:00:00).
#
# When the render job finishes, open <sample dir>/index.html in a browser (the
# tag palette is pre-seeded), tag, then "Export tags" and drop the resulting
# loma_tags.json next to it — experiments/selection_state.py reads it from
# <OUTPUT_DIR>/experiments/loma_breakdowns/PlanarTrackInitfix/loma_tags.json.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/personal/serycjon/flatsam2/results}"
N_FRAMES="${N_FRAMES:-150}"
MAX_PER_SEQ="${MAX_PER_SEQ:-4}"
SEED="${SEED:-0}"
TIME="${TIME:-08:00:00}"

LABEL="PlanarTrackInitfix"
LIST_DIR="$OUTPUT_DIR/experiments/loma_breakdowns/$LABEL"
SAMPLE_DIR="$OUTPUT_DIR/experiments/loma_breakdowns/${LABEL}_sample"

cd "$REPO_DIR"

if [[ "${1:-run}" == "status" ]]; then
    echo "list   : $LIST_DIR/breakdown_frames.csv"
    [[ -f "$LIST_DIR/breakdown_frames.csv" ]] \
        && echo "         $(( $(wc -l < "$LIST_DIR/breakdown_frames.csv") - 1 )) frames" \
        || echo "         MISSING"
    echo "sample : $SAMPLE_DIR/breakdown_frames.csv"
    [[ -f "$SAMPLE_DIR/breakdown_frames.csv" ]] \
        && echo "         $(( $(wc -l < "$SAMPLE_DIR/breakdown_frames.csv") - 1 )) frames" \
        || echo "         MISSING"
    echo "gallery: $SAMPLE_DIR/index.html"
    [[ -f "$SAMPLE_DIR/index.html" ]] && echo "         rendered" || echo "         MISSING"
    echo "tags   : $LIST_DIR/loma_tags.json"
    [[ -f "$LIST_DIR/loma_tags.json" ]] && echo "         present" || echo "         MISSING"
    exit 0
fi

echo "== 1/3 refresh the ranked breakdown list =="
python experiments/loma_breakdown_gallery.py list --output-dir "$LIST_DIR"

echo
echo "== 2/3 stratified random sample =="
python experiments/loma_breakdown_gallery.py sample \
    --list-dir "$LIST_DIR" \
    --output-dir "$SAMPLE_DIR" \
    --n-frames "$N_FRAMES" \
    --max-per-seq "$MAX_PER_SEQ" \
    --seed "$SEED"

echo
echo "== 3/3 submit the GPU render of the sample =="
# max-per-seq is enforced by the sampler already; the render must not re-cap.
python slurm/submit_cmd.py --job-name breakdown-sample --gpus 1 --time "$TIME" -- \
    python experiments/loma_breakdown_gallery.py render \
        --output-dir "$SAMPLE_DIR" \
        --max-per-seq 9999 \
        --max-total "$N_FRAMES"

echo
echo "Submitted. When it finishes: open $SAMPLE_DIR/index.html, tag, export,"
echo "and save the export as $LIST_DIR/loma_tags.json"
