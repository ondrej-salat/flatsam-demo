#!/bin/bash
# SLURM batch script for flatsam2 tracking jobs.
#
# This file is NOT meant to be edited directly.  Use slurm/submit.py to
# generate and submit jobs — it sets all FLATSAM2_* environment variables and
# calls sbatch with the right --array / resource flags.
#
# Packing: with FLATSAM2_PACK=P, array task K runs work parts [(K-1)*P, K*P)
# concurrently on one shared GPU.  A background sampler records GPU/CPU/RAM
# usage so slurm/summarize_resources.py can suggest a better --pack next time.
#
# Manual one-off usage (single sequence list file, no array):
#   FLATSAM2_RUN_ARGS="pot --output-dir ~/data/flatsam2/results --base-dir /data/POT-210" \
#   FLATSAM2_SUBSET_FILES=/path/to/mylist.txt \
#   FLATSAM2_REPO_ROOT=$(pwd) \
#   sbatch --time=4:00:00 --gpus=1 slurm/run.sbatch

#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL

set -uo pipefail
# NOTE: no `set -e` — we wait on all background pythons and aggregate their
# exit codes before exiting, rather than dying on the first failure.

# ---------------------------------------------------------------------------
# Pick the subset files for this array task.
# FLATSAM2_SUBSET_FILES is a colon-separated list of paths (one per work part).
# With FLATSAM2_PACK=P, array task K runs parts [(K-1)*P, K*P) in parallel.
# ---------------------------------------------------------------------------
IFS=':' read -ra _SUBSET_FILES <<< "${FLATSAM2_SUBSET_FILES}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
PACK="${FLATSAM2_PACK:-1}"
TOTAL=${#_SUBSET_FILES[@]}

if ! [[ "${PACK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: FLATSAM2_PACK must be a positive integer; got '${PACK}'" >&2
    exit 1
fi
if ! [[ "${TASK_ID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be a positive integer; got '${TASK_ID}'" >&2
    exit 1
fi

START=$(( (TASK_ID - 1) * PACK ))
END=$(( START + PACK ))
(( END > TOTAL )) && END=${TOTAL}

if (( START < 0 || START >= TOTAL )); then
    echo "ERROR: TASK_ID ${TASK_ID} with PACK ${PACK} maps to slot range [${START},${END}) out of [0,${TOTAL})." >&2
    exit 1
fi

SLOT_FILES=()
for (( i=START; i<END; i++ )); do
    SLOT_FILES+=("${_SUBSET_FILES[$i]}")
done

if (( ${#SLOT_FILES[@]} == 0 )); then
    echo "ERROR: empty SLOT_FILES for task ${TASK_ID} (PACK=${PACK}, TOTAL=${TOTAL}). Refusing to run a no-op." >&2
    exit 1
fi

for f in "${SLOT_FILES[@]}"; do
    if [ -z "${f}" ] || [ ! -f "${f}" ]; then
        echo "ERROR: subset file not found for task ${TASK_ID}: '${f}'" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Change to repo root, activate the environment
# ---------------------------------------------------------------------------
cd "${FLATSAM2_REPO_ROOT}"
# FLATSAM2_ENV_FILE / FLATSAM2_ENTRY let a submission swap in a different
# environment + entry point; legacy (WOFTSAM / SAM-H) runs use
# slurm/legacy_env.sh + experiments/run_legacy.py. Defaults are flatsam2's own.
ENV_FILE="${FLATSAM2_ENV_FILE:-.envrc}"
ENTRY="${FLATSAM2_ENTRY:-run.py}"
source "${ENV_FILE}"

LOG_DIR="${FLATSAM2_REPO_ROOT}/slurm/logs"
mkdir -p "${LOG_DIR}"

JOB_TAG="${SLURM_JOB_NAME:-flatsam2}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${TASK_ID}"

# ---------------------------------------------------------------------------
# Node-local scratch for the hot, concurrent log writes (slot logs + resource
# CSV).  Under big arrays, many tasks hammering the shared NFS slurm/logs dir
# triggers NFS ESTALE ("Stale file handle"): tee dies on open, the slot python
# gets SIGPIPE and dies, and pipefail marks the part FAILED — losing real work.
# /data/temporary is the HPC's per-node job storage (fast, node-only, wiped
# after the job), so logs are written there and copied back to NFS at the end.
# Fall back to SLURM_TMPDIR or /tmp if it is missing/unwritable.
# ---------------------------------------------------------------------------
LOCAL_BASE="${FLATSAM2_LOCAL_SCRATCH:-/data/temporary}"
if [ ! -d "${LOCAL_BASE}" ] || [ ! -w "${LOCAL_BASE}" ]; then
    LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
fi
# Per-user: another user's flatsam2_logs/ there is not writable for us
LOCAL_LOG_DIR="${LOCAL_BASE}/flatsam2_logs_$(id -un)/${JOB_TAG}"
mkdir -p "${LOCAL_LOG_DIR}" || LOCAL_LOG_DIR="${LOG_DIR}"
echo "Task ${TASK_ID}: node-local log dir ${LOCAL_LOG_DIR} (copied back to ${LOG_DIR} at end)"

# ---------------------------------------------------------------------------
# Background sampler — GPU + host CPU/RAM, once per interval.
# Writes a CSV that summarize_resources.py can ingest.
# ---------------------------------------------------------------------------
RES_CSV="${LOCAL_LOG_DIR}/resources_${JOB_TAG}.csv"
SAMPLE_INTERVAL="${FLATSAM2_SAMPLE_INTERVAL:-60}"
SCRIPT_PID=$$

sampler_loop() {
    local csv="$1"
    local interval="$2"
    local pgid="$3"
    echo "timestamp,gpu_mem_used_mib,gpu_mem_total_mib,gpu_util_pct,gpu_mem_util_pct,host_mem_used_mib,host_mem_total_mib,load1,proc_rss_mib,proc_pcpu_sum,proc_count" > "${csv}"
    while true; do
        local ts gpu_line mem_kb_total mem_kb_avail mem_used_mib mem_total_mib load1 ps_data rss_mib pcpu n_proc
        ts=$(date -Iseconds)
        if command -v nvidia-smi >/dev/null 2>&1; then
            gpu_line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory \
                                  --format=csv,noheader,nounits 2>/dev/null \
                       | head -1 | sed 's/, */,/g' | tr -d ' ')
        fi
        [ -z "${gpu_line:-}" ] && gpu_line=",,,"
        mem_kb_total=$(awk '/^MemTotal:/{print $2; exit}' /proc/meminfo)
        mem_kb_avail=$(awk '/^MemAvailable:/{print $2; exit}' /proc/meminfo)
        mem_used_mib=$(( (mem_kb_total - mem_kb_avail) / 1024 ))
        mem_total_mib=$(( mem_kb_total / 1024 ))
        load1=$(awk '{print $1; exit}' /proc/loadavg)
        ps_data=$(ps -o rss=,pcpu= -g "${pgid}" 2>/dev/null)
        rss_mib=$(printf '%s\n' "${ps_data}" | awk '{s+=$1} END{print int(s/1024)}')
        pcpu=$(printf '%s\n' "${ps_data}" | awk '{s+=$2} END{printf "%.1f", s+0}')
        n_proc=$(printf '%s\n' "${ps_data}" | awk 'NF>0' | wc -l)
        echo "${ts},${gpu_line},${mem_used_mib},${mem_total_mib},${load1},${rss_mib},${pcpu},${n_proc}" >> "${csv}"
        sleep "${interval}"
    done
}

sampler_loop "${RES_CSV}" "${SAMPLE_INTERVAL}" "${SCRIPT_PID}" &
SAMPLER_PID=$!

cleanup() {
    if [ -n "${SAMPLER_PID:-}" ]; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
    fi
}

# Copy node-local logs back to permanent NFS. Must run even on abnormal exit
# (timeout / OOM / scancel), otherwise the slot logs — which hold the full
# python stdout+stderr, including swallowed tracker tracebacks — are wiped with
# the node and the failure becomes uninvestigable. Idempotent; safe to re-run.
copy_logs_back() {
    [ -z "${LOCAL_LOG_DIR:-}" ] && return 0
    [ "${LOCAL_LOG_DIR}" = "${LOG_DIR:-}" ] && return 0
    cp -f "${LOCAL_LOG_DIR}/${JOB_TAG}_slot"*.log "${LOG_DIR}/" 2>/dev/null || true
    [ -n "${RES_CSV:-}" ] && cp -f "${RES_CSV}" "${LOG_DIR}/" 2>/dev/null || true
}
trap 'cleanup; copy_logs_back' EXIT INT TERM

# ---------------------------------------------------------------------------
# Launch PACK python invocations concurrently, each with its own log file.
# FLATSAM2_RUN_ARGS is a shell-quoted run.py argument string built by submit.py;
# we append --subset-file per slot since the path is resolved per work part.
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1

# Divide the task's CPU allocation across the packed slots so the parallel
# pythons don't oversubscribe BLAS/OMP thread pools.
CPUS_TOTAL="${SLURM_CPUS_PER_TASK:-$(nproc)}"
N_SLOTS=${#SLOT_FILES[@]}
PER_SLOT_THREADS=$(( CPUS_TOTAL / N_SLOTS ))
(( PER_SLOT_THREADS < 1 )) && PER_SLOT_THREADS=1
export OMP_NUM_THREADS="${PER_SLOT_THREADS}"
export MKL_NUM_THREADS="${PER_SLOT_THREADS}"
export OPENBLAS_NUM_THREADS="${PER_SLOT_THREADS}"
export NUMEXPR_NUM_THREADS="${PER_SLOT_THREADS}"
echo "Task ${TASK_ID}: ${CPUS_TOTAL} CPU(s) / ${N_SLOTS} slot(s) → ${PER_SLOT_THREADS} thread(s) per slot"

read -ra BASE_ARGS <<< "${FLATSAM2_RUN_ARGS}"

PIDS=()
SLOT_LABELS=()
for slot in "${!SLOT_FILES[@]}"; do
    subset="${SLOT_FILES[$slot]}"
    abs_slot=$(( START + slot + 1 ))   # 1-based global part index, for humans
    slot_log="${LOCAL_LOG_DIR}/${JOB_TAG}_slot${abs_slot}.log"
    SLOT_LABELS+=("part${abs_slot}")

    args=("${BASE_ARGS[@]}" "--subset-file" "${subset}")
    echo "Task ${TASK_ID} slot ${abs_slot}: ${ENTRY} ${args[*]}  (log: ${slot_log})"

    (
        set -o pipefail
        python -u "${ENTRY}" "${args[@]}" 2>&1 \
            | tee "${slot_log}" \
            | awk -v tag="[T${TASK_ID}/S${abs_slot}]" \
                '/^Sequence: |^Sequence time:|^Elapsed:|^Skipping |^Processing |ERROR/ {
                    print tag, $0; fflush()
                }'
    ) &
    PIDS+=($!)
done

# Wait on every child; remember the worst exit status.
EXIT_STATUS=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    label="${SLOT_LABELS[$idx]}"
    if wait "${pid}"; then
        echo "Task ${TASK_ID} ${label} (pid ${pid}): OK"
    else
        rc=$?
        echo "Task ${TASK_ID} ${label} (pid ${pid}): FAILED with exit ${rc}" >&2
        EXIT_STATUS=${rc}
    fi
done

# ---------------------------------------------------------------------------
# Stop the sampler now so its CSV is closed before we summarize it.
# ---------------------------------------------------------------------------
cleanup
# Sampler already stopped; keep copying logs back on any later exit/kill.
trap copy_logs_back EXIT INT TERM
SAMPLER_PID=""

# ---------------------------------------------------------------------------
# Per-task resource summary printed into this task's slurm .out log.
# Pass --pack and --mail-to so summarize_resources.py can email when the pack
# looks misconfigured (GPU memory over the limit, or badly under-utilised).
# Cross-task aggregation later with:
#   python slurm/summarize_resources.py 'slurm/logs/resources_<jobname>_*.csv'
# ---------------------------------------------------------------------------
echo
echo "=========================================================================="
echo "Resource usage summary (task ${TASK_ID}, pack=${PACK})"
echo "  CSV: ${RES_CSV}"
echo "=========================================================================="
if [ -s "${RES_CSV}" ]; then
    python "${FLATSAM2_REPO_ROOT}/slurm/summarize_resources.py" "${RES_CSV}" \
        --pack "${PACK}" \
        --mail-to "${FLATSAM2_MAIL_TO:-}" \
        --job-tag "${JOB_TAG}" \
        || echo "(summarize_resources.py failed; CSV still on disk)"
else
    echo "(no samples collected — task may have been too short)"
fi
echo "=========================================================================="

# ---------------------------------------------------------------------------
# Copy node-local logs back to the shared NFS log dir for later inspection.
# /data/temporary is node-only and wiped after the job, so this must happen
# before we exit.  Failures here are non-fatal — the compute already ran, and
# we don't want a flaky NFS copy to mask the real EXIT_STATUS.
# ---------------------------------------------------------------------------
if [ "${LOCAL_LOG_DIR}" != "${LOG_DIR}" ]; then
    cp -f "${LOCAL_LOG_DIR}/${JOB_TAG}_slot"*.log "${LOG_DIR}/" 2>/dev/null || true
    cp -f "${RES_CSV}" "${LOG_DIR}/" 2>/dev/null || true
fi

exit ${EXIT_STATUS}
