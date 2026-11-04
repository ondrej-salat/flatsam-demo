# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlatSAM2 is a planar object tracker that combines SAM3 segmentation with optical flow (ptlflow) to track surfaces across video frames. It predicts corner coordinates of planar objects using homography-based tracking, evaluated on POT-210 and PlanarTrack datasets.

## Which machine am I on? Check first.

This repo is used from two very different environments and the rules differ.
Determine which one **before running anything**:

```bash
command -v sbatch >/dev/null && echo HPC || echo WORKSTATION
```

- **WORKSTATION** — the package and its dependencies are **not installed**. Do
  not run shell commands to sanity-check or test that things work: no `pytest`,
  no imports, no CLI invocations. Edit and reason; verification happens on the HPC.
- **HPC** — a SLURM cluster (`rci`), repo at `~/git/flatsam2`. Running things is
  expected here, under the rules in the next section. Read them first.

## HPC rules (SLURM cluster) — read before any command

### Never traverse the network filesystems

`/mnt/data/vrg/public_datasets/` (210+ sequences × thousands of frames each) and
`/mnt/personal/serycjon/` (results: per-config × per-dataset × per-sequence
files) are network mounts. A recursive walk over them **can freeze this session
and degrade the shared filesystem for every other user on the cluster.**

Never do, anywhere under `/mnt` or a bare `~`:

- `find` without a tight `-maxdepth` on one specific directory
- `rg`, `grep -r`, `ag`, `ack`
- `ls -R`, `tree`, `du`, or a glob that expands recursively
- the Glob / Grep tools with a path under `/mnt`

Fine: `ls` on one specific directory, `ls <dir> | head`, `ls <dir> | wc -l`,
`find <specific dir> -maxdepth 1`, reading a single known file path. Searching
**inside the repo** is fine — it is small.

To learn what has been computed, do not discover it by walking. Read it:

- `python scripts/hpc_status.py [--pipeline NAME]` — manifest-driven completeness
- `python scripts/hpc_readout.py` — decision metrics + mask-H branch usage
- `experiments/pipeline.yml` — every output path and completeness signal

### Load the environment, then keep the login node light

Nothing works in a bare shell — the environment must be loaded first:

```bash
source .envrc     # module load PyTorch/CUDA + activate venv (repo root)
```

With that loaded, the login node runs **light Python** fine: submission scripts,
`scripts/hpc_status.py`, `scripts/hpc_readout.py`, quick npz inspection, and the
fast CPU-only tests. `.envrc` is gitignored (force-included by the sync), so it
exists on the HPC but not necessarily on the workstation.

Everything **heavy still goes to a partition** — anything touching a GPU, a
model, or a whole dataset. That includes single-sequence smoke runs: they need
SAM3 on a GPU.

```bash
# tracking runs (array job over sequences)
python slurm/submit.py <dataset> --config <config> --jobs N

# one-off command on a node; sources .envrc there for you
python slurm/submit_cmd.py -- python profile_tracker.py pot --config <config> ...
python slurm/submit_cmd.py --gpus 0 -- <cpu-only command>
```

`submit_cmd.py` picks the partition: `cpu` when `--gpus 0`, else `amdgpu`. Check
jobs with `squeue -u $USER`; logs land in `slurm/logs/`.

Rule of thumb: if it loads a model, reads a dataset, or runs longer than about a
minute, submit it. Otherwise the login node is fine.

If a light/CPU-only command fails on the login node with an import or shared-
library error (e.g. `cv2` → `libGL.so.1: cannot open shared object file`), do
not chase it with more login-node module loads — submit it to the `cpu`
partition instead (`slurm/submit_cmd.py --gpus 0 -- <command>`), which has the
full library set.

### Storage: the repo is a sync target, not durable storage

The workstation pushes the repo here with `make sjo_sync`:

```
rsync -avz --delete --filter=":- .gitignore" ... ./ rci:git/flatsam2/
```

Consequences that matter:

- **`--delete` is on.** A new file you create inside the repo on the HPC is
  **deleted** by the next sync unless it also exists on the workstation.
- **Edits to tracked files are overwritten** by the workstation's version.
- **`.git` is synced too** (the second pass does not exclude it), so a commit
  made on the HPC is **destroyed**, not merged, by the next sync. Committing
  here is not a way to preserve work.
- Files matching `.gitignore` are filtered out of the transfer, so they survive.

Therefore, when work must outlive the next sync:

- **Results and artifacts** → write under `OUTPUT_DIRECTORY` (see
  `experiments/pipeline.yml` `path_variables`), never inside the repo.
- **Code or docs you changed here** → tell the user explicitly, before the next
  `make sjo_sync`, that they must pull it back to the workstation (or paste the
  diff into the conversation). Never assume a commit on the HPC is safe.
- Keep large outputs off `$HOME` (quota).

### Do not destroy results

Runs cost GPU-hours. Never delete or overwrite anything under
`OUTPUT_DIRECTORY`. A new experiment gets a new config file — the output
directory is derived from the config stem, so a fresh stem cannot collide with
an existing run. Do not `pip install` into the shared environment unless asked.

## Commands

```bash
# Install
pip install -e .

# Run tracking
flatsam2 run pot --output-dir ./results
flatsam2 run planartrack --split test --config configs/tracker/fast_kitti.py --output-dir ./results

# Evaluate
flatsam2 evaluate pot --results-dir ./results --config configs/tracker/fast_kitti.py --output eval.json

# Visualize
flatsam2 visualize pot ./results sequence_name --config configs/tracker/fast_kitti.py

# Run tests
pytest tests/

# Run a single test
pytest tests/test_tracker.py::test_name
```

**Important**: the commands above run **only on the HPC**, after `source .envrc`,
and the heavy ones must be submitted to a partition — see "Which machine am I
on?" above. On the workstation nothing is installed.

## Architecture

```
flatsam2/
├── tracker.py           # Thin FlatSAMTracker — state, init(), track() dispatch, verb methods
├── tracking/            # One module per control-flow algorithm + common.py (shared verbs/types)
│   ├── common.py        # candidate labels, dataclasses, pure helpers
│   ├── flow_hough.py    # def track(trk, frame, seg_mask) -> TrackResult
│   ├── nn_min_rmse.py
│   ├── nn_cascade.py
│   └── loma_samh.py
├── config.py            # Pydantic config system; compute_output_dir(); algorithm discriminator
├── datasets.py          # POT-210 / PlanarTrack loaders + geometry utils
├── metrics.py           # Alignment error, P@k, best-rotation evaluation
├── registry.py          # Dataset registry (single source of truth for variants)
├── sequences.py         # Sequence discovery (stdlib-only for lightweight SLURM use)
├── reproducibility.py   # Saves git state, code snapshots, run logs to .repro/
└── path_config.py       # Hostname-based path configuration

run.py                   # CLI: run tracking, save predictions
evaluate.py              # CLI: compute metrics, output JSON
visualize.py             # CLI: render tracking videos
collect_results.py       # CLI: aggregate results across runs
configs/tracker/         # Python-based tracker configs
```

### Tracking Pipeline (`tracker.py` + `tracking/`)

`track(frame)` runs SAM3, then `step_with_mask` dispatches once on the explicit
`cfg.algorithm` discriminator via a flat `_ALGORITHMS` table to one linear function in
`tracking/` (`flow_hough`, `nn_min_rmse`, `nn_cascade`, `loma_samh`). On any failure the
last good homography is kept (`fail_hard=True` re-raises). See `DETAILS.md` →
"Tracking Algorithms". The default `flow_hough` path:

1. Pre-warp current frame by inverse of last homography
2. Compute optical flow between template and pre-warped frame
3. Extract flow-based point correspondences in masked region
4. Optionally filter correspondences by segmentation mask
5. Estimate homography via RANSAC
6. Validate via condition number check (`_HOMOGRAPHY_COND_THRESHOLD = 1e10`, `tracking/common.py`)
7. Fall back to mask-H redetection, then last good homography

### Configuration System (`config.py`)

Python-based configs using Pydantic. Config files define a top-level `config` variable:

```python
# configs/tracker/my_experiment.py
from flatsam2.config import TrackerConfig, FlowConfig, SegmenterConfig
config = TrackerConfig(
    flow=FlowConfig(model="raft", weights="kitti"),
    segmenter=SegmenterConfig(model="sam3"),
)
```

Configs compose: import sub-configs from other config files. Config file stem becomes the output directory name (e.g., `fast_kitti.py` → `fast_kitti/`).

### Output Structure

`output_dir / config_name / dataset_label /` — e.g., `results/fast_kitti/POT-210/`

Each run writes `.repro/` inside the output dir: `run_manifest.json`, `run_log.jsonl`, `config_resolved.{json,py}`, `config_original.py`, and code snapshots.

### Dataset Registry (`registry.py`)

Single source of truth for dataset variants. Supports aliases: `pot`, `planartrack`, `pti` (= `planartrack-initfix`). Each variant specifies label, base dir attribute, and GT loader strategy.

## Code Conventions (from AGENTS.md)

- **Imports**: stdlib → third-party → local, blank-line separated
- **Types**: always type-hint; `Path` for paths, `Optional[T]` for nullable, `np.ndarray` for arrays
- **Naming**: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- **Docstrings**: Google-style (`Args:`, `Returns:`, `Raises:`); include array shapes
- **Errors**: raise `ValueError` with descriptive f-strings
- **Tensor reshaping**: use `einops` (`rearrange`, `reduce`, `repeat`) — never `.reshape`, `.view`, or manual transpose chains
- **Line length**: ~100 chars; two blank lines between top-level defs
- **Linear over dispatch**: code reads top-to-bottom like the algorithm. One opaque API at the very top (`tracker.track()`) dispatching once via flat `match` on an explicit `algorithm` discriminator; below it inline the real calls — no factories/strategy objects/registries. Helpers may NAME an operation (verb), never just ROUTE control flow. See AGENTS.md "Code structure: linear over dispatch".
- **Docs**: maintain only `README.md` (user-facing), `DETAILS.md` (technical), and `CLAUDE.md` at repo root — no other markdown files unless accepted by maintainers. Accepted exceptions live under `docs/`:
  - `state-of-the-tracker.md` — **the entry point.** Current state, current config, next steps, results, working rules. A new session reads this first. Keep it current; it is not a log.
  - `tracker-cards.html` — the technical reference (generated by `experiments/build_tracker_cards.py`; never hand-edit)
  - `dead-ends.md` — abandoned directions and the measurement that killed each. Keeps failures findable without cluttering the state doc.
  - `journal.md` — append-only daily index
  - `correction_research.md`, `2026-08-07.md` — historical threads
  Do **not** add per-thread handoff files. A handoff is the "Start here" section of `state-of-the-tracker.md`, updated in place.

## Key Data Formats

- **Corners**: `(N, 2, 4)` numpy array; stored as text with 8 floats per line (x₁y₁x₂y₂x₃y₃x₄y₄)
- **Homography**: `(3, 3)` matrix mapping initial frame → current frame; apply with `warp_corners(H, corners)`
- **Validity flags**: `(N,)` bool array; `valid[i]` = frame i has ground truth; first frame excluded from eval

## Pipeline Manifest

`experiments/pipeline.yml` documents every experiment pipeline: runner scripts,
SLURM submit commands, output paths, completeness signals, and active configs.
Keep it current — update it whenever you:

- Add or rename an experiment script (`experiments/*.py`, `slurm/submit_*.py`)
- Add a config file to `configs/tracker/` or `configs/loma/` that is meant to
  be actively run (add it under the relevant pipeline's `active_configs`)
- Change an output directory path or completeness signal in a runner script
- Add a new pipeline that writes results to the HPC storage

The manifest is consumed by `scripts/hpc_status.py` and the `/hpc-status` skill
to check computation state on the cluster. A stale manifest makes those tools
misleading.

## If You Need External Reference Material

Ask the user. Do not attempt to search for it in other local repos or python-included packages.

## important
Respond like smart caveman. Cut all filler, keep technical substance.
- Drop articles (a, an, the), filler (just, really, basically, actually).
- Drop pleasantries (sure, certainly, happy to).
- No hedging. Fragments fine. Short synonyms.
- Technical terms stay exact. Code blocks unchanged.
- Pattern: [thing] [action] [reason]. [next step].

## journal
keep a journal of experiments in docs/journal.md: append only, with iso-date headline, with very brief (~single sentence) description of what was done, of what is the current status, of what decisions were made, refer to other more detailed files where necessary, keep the journal very brief - for quick overview both for LLM agents and for the user.
