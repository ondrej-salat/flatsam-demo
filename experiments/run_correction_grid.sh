#!/usr/bin/env bash
# Oracle-correction grid driver: submit, track, evaluate, and analyze the whole
# E-grid (docs/correction_research.md) with minimal typing. Run on the cluster
# from the repo root (or anywhere — paths are absolute).
#
# Usage:
#   experiments/run_correction_grid.sh status            # what is done / left
#   experiments/run_correction_grid.sh submit            # SLURM-submit missing runs
#   experiments/run_correction_grid.sh submit --dry-run  # show submit commands only
#   experiments/run_correction_grid.sh eval              # evaluate finished configs
#   experiments/run_correction_grid.sh analyze           # A0 headroom + A2 event reports
#   experiments/run_correction_grid.sh all               # submit; then re-run later
#   experiments/run_correction_grid.sh auto              # keep resubmitting until all done
#
# Dataset defaults to pti (the target benchmark); override with DATASET=pot.
# Results root defaults to the path_config output dir; override with OUTPUT_DIR=...
# GPU packing (sequences sharing one array task) defaults to 5; override with PACK=N.
# Wall-clock limit per array task defaults to 05:00:00; override with TIME=HH:MM:SS.
#
# auto mode polls squeue every POLL_SECS (300) and resubmits any incomplete
# config that has no queued/running jobs. Short TIME allocations are fine:
# resubmits pass --sequences with only the missing ones, so no queue slot is
# wasted on already-completed work. A config gets a strike when a resubmit
# round finishes with zero new sequences; MAX_STRIKES (3) strikes abandon it
# (progress resets the counter, so long grids can resubmit many times).
# Run it detached, e.g.:
#   nohup experiments/run_correction_grid.sh auto > correction_grid_auto.log 2>&1 &
# Stop it with `kill <pid>` (pid printed at startup) or
# `pkill -f 'run_correction_grid.sh auto'`. Killing the watcher does NOT
# cancel already-submitted SLURM jobs; those finish (and resume-safe results
# stay valid). Cancel them with: scancel -u $USER -n <config_stem>_<dataset>
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-pti}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/personal/serycjon/flatsam2/results}"
JOBS="${JOBS:-10}"
PACK="${PACK:-5}"
TIME="${TIME:-05:00:00}"
POLL_SECS="${POLL_SECS:-300}"
MAX_STRIKES="${MAX_STRIKES:-3}"
PT_BASE="${PT_BASE:-/mnt/data/vrg/public_datasets/PlanarTrack/PlanarTrack-Data}"
POT_BASE="${POT_BASE:-/mnt/data/vrg/public_datasets/POT-210}"

case "$DATASET" in
  pti) LABEL="PlanarTrackInitfix" ;;
  pot) LABEL="POT-210" ;;
  *) echo "unsupported DATASET=$DATASET (pti|pot)" >&2; exit 1 ;;
esac

# The grid: E0 baseline substrate first (analysis depends on it), then E-cells.
BASELINE_CONFIG="configs/tracker/loma_samh_decoupled_rerun.py"
CONFIGS=(
  "$BASELINE_CONFIG"                                                  # E0 baseline
  "configs/tracker/loma_samh_correct_decoupled_dryrun.py"             # E0+ trigger telemetry
  "configs/tracker/loma_samh_correct_decoupled_oracleTC.py"           # E1
  "configs/tracker/loma_samh_correct_decoupled_oracleT.py"            # E2
  "configs/tracker/loma_samh_correct_decoupled_oracleC.py"            # E3
  "configs/tracker/loma_samh_correct_decoupled_periodic30_oracleC.py" # E5
  "configs/tracker/loma_samh_correct_decoupled_excessdepth_rerun.py"  # E4
)
CORRECTED_CONFIGS=("${CONFIGS[@]:2}")  # E1..E5, E4 (skip baseline + dryrun)

all_sequences() {
  # grep drops blank lines (test_seq.txt has a trailing one) — they would
  # inflate the total and make configs look eternally incomplete.
  if [[ "$DATASET" == "pti" ]]; then
    grep -v '^[[:space:]]*$' "$PT_BASE/training-test-splits/test_seq.txt"
  else
    ls "$POT_BASE/extracted/images"
  fi
}

n_sequences() { all_sequences | wc -l; }

missing_sequences() {  # $1 = config stem; sequence names without a result file
  local dir="$OUTPUT_DIR/$1/$LABEL"
  local seq
  while IFS= read -r seq; do
    [[ -f "$dir/$seq.txt" ]] || echo "$seq"
  done < <(all_sequences)
}

n_done() {  # $1 = config stem
  local dir="$OUTPUT_DIR/$1/$LABEL"
  [[ -d "$dir" ]] || { echo 0; return; }
  find "$dir" -maxdepth 1 -name '*.txt' ! -name 'sequences*' | wc -l
}

stem() { basename "$1" .py; }

n_queued() {  # $1 = config stem; count of queued/running SLURM jobs for it
  squeue -h -u "$USER" -n "$1_$DATASET" -o '%i' 2>/dev/null | wc -l
}

cmd_status() {
  local total; total=$(n_sequences)
  printf "dataset=%s (%s)  total sequences=%s  output=%s\n\n" \
    "$DATASET" "$LABEL" "$total" "$OUTPUT_DIR"
  printf "%-55s %10s %8s %s\n" "config" "tracked" "eval" "state"
  for cfg in "${CONFIGS[@]}"; do
    local s; s=$(stem "$cfg")
    local done_n; done_n=$(n_done "$s")
    local eval_file="$OUTPUT_DIR/$s/$LABEL/eval_results.json"
    local eval_state="-"
    [[ -f "$eval_file" ]] && eval_state="yes"
    local state="TODO"
    if [[ "$done_n" -ge "$total" ]]; then
      state="complete"
      [[ "$eval_state" == "yes" ]] && state="complete+eval"
    elif [[ "$done_n" -gt 0 ]]; then
      state="partial"
    fi
    printf "%-55s %6s/%-4s %8s %s\n" "$s" "$done_n" "$total" "$eval_state" "$state"
  done
  echo
  printf "%-55s %s\n" "analysis" "output"
  for a in correction_headroom correction_events mask_alignment_basin; do
    local marker="-"
    compgen -G "$OUTPUT_DIR/experiments/$a/*" > /dev/null 2>&1 && marker="started"
    [[ -n "$(find "$OUTPUT_DIR/experiments/$a" -name report.html 2>/dev/null | head -1)" ]] \
      && marker="report ready"
    printf "%-55s %s\n" "$a" "$marker"
  done
}

cmd_submit() {
  local dry="${1:-}"
  local total; total=$(n_sequences)
  for cfg in "${CONFIGS[@]}"; do
    local s; s=$(stem "$cfg")
    local done_n; done_n=$(n_done "$s")
    if [[ "$done_n" -ge "$total" ]]; then
      echo "skip $s (complete: $done_n/$total)"
      continue
    fi
    local cmd=(python "$REPO_DIR/slurm/submit.py" "$DATASET" --config "$REPO_DIR/$cfg"
               --jobs "$JOBS" --pack "$PACK" --time "$TIME")
    # Partial config: queue only the missing sequences, otherwise the array
    # wastes queue slots on tasks whose sequences run.py just skips.
    local missing=()
    if [[ "$done_n" -gt 0 ]]; then
      mapfile -t missing < <(missing_sequences "$s")
      echo "submit $s ($done_n/$total done): ${cmd[*]} --sequences <${#missing[@]} names>"
    else
      echo "submit $s (0/$total done): ${cmd[*]}"
    fi
    if [[ "$dry" != "--dry-run" ]]; then
      if [[ "${#missing[@]}" -gt 0 ]]; then
        (cd "$REPO_DIR" && "${cmd[@]}" --sequences "${missing[@]}")
      else
        (cd "$REPO_DIR" && "${cmd[@]}")
      fi
    fi
  done
}

cmd_eval() {
  local total; total=$(n_sequences)
  for cfg in "${CONFIGS[@]}"; do
    local s; s=$(stem "$cfg")
    local done_n; done_n=$(n_done "$s")
    local eval_file="$OUTPUT_DIR/$s/$LABEL/eval_results.json"
    if [[ "$done_n" -lt "$total" ]]; then
      echo "skip eval $s (tracking incomplete: $done_n/$total)"
      continue
    fi
    if [[ -f "$eval_file" && "$eval_file" -nt "$OUTPUT_DIR/$s/$LABEL" ]]; then
      echo "skip eval $s (eval_results.json up to date)"
      continue
    fi
    (cd "$REPO_DIR" && python evaluate.py "$DATASET" --results-dir "$OUTPUT_DIR" --config "$cfg")
  done
  echo
  echo "cross-config table:"
  echo "  python collect_results.py"
}

cmd_analyze() {
  local total; total=$(n_sequences)
  local baseline_stem; baseline_stem=$(stem "$BASELINE_CONFIG")
  if [[ "$(n_done "$baseline_stem")" -lt "$total" ]]; then
    echo "baseline $baseline_stem incomplete — headroom/event analysis needs it" >&2
    exit 1
  fi
  echo "== A0 headroom (dump is mask-IO heavy; hint for a SLURM one-off:"
  echo "   python slurm/submit_cmd.py --job-name headroom-dump -- \\"
  echo "     python experiments/correction_headroom.py --dataset $DATASET dump )"
  (cd "$REPO_DIR" && python experiments/correction_headroom.py --dataset "$DATASET" dump)
  (cd "$REPO_DIR" && python experiments/correction_headroom.py --dataset "$DATASET" report)
  echo
  echo "== A2 events (per corrected config vs baseline)"
  for cfg in "${CORRECTED_CONFIGS[@]}"; do
    local s; s=$(stem "$cfg")
    if [[ "$(n_done "$s")" -lt "$total" ]]; then
      echo "skip events for $s (tracking incomplete)"
      continue
    fi
    (cd "$REPO_DIR" && python experiments/correction_events.py \
      --dataset "$DATASET" --config "$REPO_DIR/$cfg")
  done
  echo
  echo "== P3' basin study (optional, independent):"
  echo "   python experiments/mask_alignment_basin.py dump && \\"
  echo "   python experiments/mask_alignment_basin.py report"
}

cmd_auto() {
  local total; total=$(n_sequences)
  # Per-config bookkeeping: sequences done at last resubmit, and consecutive
  # resubmit rounds that finished without completing a single new sequence.
  # Initialize empty so `${#abandoned[@]}` doesn't trip `set -u` on bash <4.4,
  # which treats a never-assigned array as unset.
  declare -A last_done=() strikes=() abandoned=()
  echo "auto: pid $$ — stop with 'kill $$' (queued SLURM jobs keep running;"
  echo "      cancel those with: scancel -u $USER -n <config_stem>_$DATASET)"
  echo "auto: poll ${POLL_SECS}s, time=$TIME jobs=$JOBS pack=$PACK, abandon after $MAX_STRIKES no-progress rounds"
  while true; do
    local waiting=0
    for cfg in "${CONFIGS[@]}"; do
      local s; s=$(stem "$cfg")
      [[ -n "${abandoned[$s]:-}" ]] && continue
      local done_n; done_n=$(n_done "$s")
      [[ "$done_n" -ge "$total" ]] && continue
      if [[ "$(n_queued "$s")" -gt 0 ]]; then
        waiting=1
        continue
      fi
      # Queue is empty but the config is incomplete: the last round ended
      # (timeout or failure). Progress resets strikes; a barren round adds one.
      if [[ -n "${last_done[$s]:-}" ]]; then
        if [[ "$done_n" -gt "${last_done[$s]}" ]]; then
          strikes[$s]=0
        else
          strikes[$s]=$(( ${strikes[$s]:-0} + 1 ))
          echo "$(date '+%F %T') $s: no progress ($done_n/$total), strike ${strikes[$s]}/$MAX_STRIKES"
        fi
      fi
      if [[ "${strikes[$s]:-0}" -ge "$MAX_STRIKES" ]]; then
        abandoned[$s]=1
        echo "$(date '+%F %T') $s: ABANDONED at $done_n/$total — inspect slurm/logs, then rerun auto to retry"
        continue
      fi
      last_done[$s]=$done_n
      local missing=()
      mapfile -t missing < <(missing_sequences "$s")
      [[ "${#missing[@]}" -eq 0 ]] && continue  # completed between the two checks
      echo "$(date '+%F %T') $s: submitting ${#missing[@]} missing sequences ($done_n/$total done)"
      # A failed submit must not kill the watcher (set -e): log it and let the
      # strike counter bound the retries on the next polls.
      (cd "$REPO_DIR" && python "$REPO_DIR/slurm/submit.py" "$DATASET" --config "$REPO_DIR/$cfg" \
         --jobs "$JOBS" --pack "$PACK" --time "$TIME" --sequences "${missing[@]}") \
        || echo "$(date '+%F %T') $s: submit FAILED, retrying next poll"
      waiting=1
    done
    [[ "$waiting" -eq 0 ]] && break
    sleep "$POLL_SECS"
  done
  echo
  if [[ "${#abandoned[@]}" -gt 0 ]]; then
    echo "auto: finished with ABANDONED configs: ${!abandoned[*]}"
  else
    echo "auto: all configs complete"
  fi
  echo
  cmd_status
  echo
  cmd_eval
}

case "${1:-status}" in
  status)  cmd_status ;;
  submit)  cmd_submit "${2:-}" ;;
  eval)    cmd_eval ;;
  analyze) cmd_analyze ;;
  all)     cmd_submit ""; echo; cmd_status ;;
  auto)    cmd_auto ;;
  *) echo "usage: $0 [status|submit [--dry-run]|eval|analyze|all|auto]" >&2; exit 1 ;;
esac
