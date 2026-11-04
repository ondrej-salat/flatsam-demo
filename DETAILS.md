# Technical Details

Technical reference for FlatSAM2 implementation and interfaces.

## Architecture

- `flatsam2/tracker.py` — Thin `FlatSAMTracker`: state, `init()`, `track()`/`step_with_mask()`, and the shared verb methods; dispatches per frame to one algorithm (see **Tracking Algorithms**)
- `flatsam2/tracking/` — One module per control-flow algorithm + `common.py` (shared verbs/types)
- `flatsam2/datasets.py` — POT-210 and PlanarTrack loaders; corner/mask utilities
- `flatsam2/metrics.py` — Alignment error, P@k, best-rotation evaluation
- `flatsam2/config.py` — Configuration system with Pydantic; `compute_output_dir()` for structured paths
- `flatsam2/run_tracking.py` — Experiment runner; saves predictions as text files
- `flatsam2/evaluate.py` — Loads results, computes metrics, outputs JSON
- `flatsam2/visualize.py` — Renders tracking videos with overlays

## Directory Structure

FlatSAM2 uses a hierarchical output structure: `base_output_dir / config_name / dataset_name /`

**Example:**
```
results/
├── default/
│   ├── POT-210/
│   │   ├── sequence1.txt
│   │   └── sequence2.txt
│   └── PlanarTrack/
│       └── ...
└── fast_kitti/
    ├── POT-210/
    └── PlanarTrack/
```

**Usage:**
- `--output-dir` specifies the **top-level** directory
- Config file stem determines subdirectory (e.g., `configs/tracker/fast_kitti.py` → `fast_kitti/`)
- If no config specified, uses `default/`
- Dataset automatically determines final level (`POT-210/` or `PlanarTrack/`)

**Utilities:**
```python
from flatsam2.config import compute_output_dir
from pathlib import Path

# Compute output path
output_dir = compute_output_dir(
    base_output_dir=Path('./results'),
    dataset='pot',
    config_path=Path('configs/tracker/fast_kitti.py')  # or None for 'default'
)
# Returns: Path('./results/fast_kitti/POT-210')
```

## Reproducibility Tracking

Each experiment run automatically saves reproducibility info into a `.repro/` subdirectory inside the output directory (e.g. `results/default/POT-210/.repro/`).

**Contents:**

| File | Purpose |
|---|---|
| `run_manifest.json` | Golden record from the first invocation: config, argv, git commit + diff, Python/platform info |
| `run_log.jsonl` | One JSON line appended per invocation (including restarts), capturing any code changes between runs |
| `config_resolved.json` | Fully resolved config with all subconfigs expanded (for quick inspection) |
| `config_resolved.py` | Re-runnable Python config file (pass to `--config` to reproduce) |
| `config_original.py` | Verbatim copy of the config file used |
| `code_snapshot_initial.tar.gz` | Tarball of repo source at first run |
| `code_snapshot_<timestamp>.tar.gz` | Additional snapshots on restart (if code changed) |

On restart (computation interrupted and resumed), a new entry is appended to `run_log.jsonl` and a fresh code snapshot is saved with a timestamped name. This ensures every code state that contributed to the results is recorded, even if the code was modified between runs.

## Data Formats

**Corners** — stored as text (N×8: x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄); processed as `(N, 2, 4)` or `(2, 4)` numpy arrays.

**Homography** — `(3, 3)` matrix mapping initial→current frame. Apply with `warp_corners(H, corners)`.

**Validity flags** — `(N,)` bool array; `valid[i]` indicates frame i has ground truth. First frame (init) excluded from eval.

## Tracker Interface

```python
# Init with corners (benchmarks)
tracker.init(frame, init_corners=corners)  # corners: (2,4)

# Init with mask (custom shapes)
tracker.init(frame, init_mask=mask)  # mask: (H,W) uint8

# Track
H = tracker.track(frame)  # returns (3,3) homography
```

## Tracking Algorithms

The tracker reads top-to-bottom like the algorithm it runs ("linear over dispatch",
see `AGENTS.md`). `track(frame)` runs SAM3, then `step_with_mask` dispatches once on an
explicit `cfg.algorithm` discriminator through a flat table — no factories/registries:

```python
# flatsam2/tracker.py
_ALGORITHMS = {
    "flow_hough":  flow_hough.track,
    "nn_min_rmse": nn_min_rmse.track,
    "nn_cascade":  nn_cascade.track,
    "loma_samh":   loma_samh.track,
}
# step_with_mask(): return _ALGORITHMS[self.cfg.algorithm](self, frame, seg_mask)
```

Each algorithm is one linear free function `track(trk, frame, seg_mask) -> TrackResult`
in `flatsam2/tracking/`, calling the tracker's verb methods (`_estimate_flow_H`,
`_compute_mask_h`, `_estimate_loma_H`, `_score_homography`, `_choose_loma_or_mask`,
`_maybe_correct_sam3`, `_run_symmetry`, `_finalize_*`) inline. The ~24 configs collapse
to 4 control-flow algorithms; everything else is parameter tweaks.

| `algorithm`   | configs (examples)                                  | path |
|---------------|-----------------------------------------------------|------|
| `flow_hough`  | `fast_kitti`, `flow_hough*`, `hough_symmetry_only*` | flow (last-H prewarp) → mask-H redetection → mask-H; symmetry on every winner |
| `nn_min_rmse` | `*_nn` (no `trust_threshold`)                       | score all candidates, pick min predicted RMSE ≤ `rmse_threshold` |
| `nn_cascade`  | `*_nn_cascade*`                                      | flow → redetection → mask-H; first ≤ `trust_threshold` wins, else best ≤ `rmse_threshold` |
| `loma_samh`   | `loma_samh*`                                         | LoMa keypoint H vs SAM-H, picked by inlier count; optional SAM3 self-correction |

`cfg.algorithm` is set explicitly in each config and cross-checked at load time against
a derivation from the legacy flags (`config._derive_algorithm`/`_resolve_algorithm`): a
mismatch raises. `loma_samh` has two inline sub-variants on `loma.decouple_hough_state`:
coupled (mask-H on shared state; symmetry gated on a mask-H win) and decoupled (an
independent SAM-H sub-tracker owns its own mask-H + symmetry state; the LoMa estimate is
picked against it per frame, nothing feeds back).

### SAM3 candidate-mask dump (segmentation only)

`config.candidate_dump=True` (see `configs/tracker/sam3_candidates.py`) is not a tracking
algorithm — it bypasses `step_with_mask` entirely. `run.py` instead streams SAM3 over the
sequence via `FlatSAMTracker.track_candidates(frame)`, which calls
`SAM3StreamingTracker.track(frame, return_all_masks=True)` and returns the selected mask
plus all candidates (`{"masks": (M,H,W) bool, "logits": (M,H,W) f32, "ious": (M,) f32}`,
`M`≤3, 0 when the object is absent). Pair with `flow_enabled=False` so no flow model is
built. Frame 0 is prompted with the GT-corners mask (same as normal runs).

Per sequence, under `results/sam3_candidates/<dataset>/masks/<seq>/`:
`{frame+1:05d}.png` (selected mask), `{frame+1:05d}_c{k}.png` (each candidate), and
`ious.json` (per-frame candidate count, predicted IoUs, best index; written last =
completeness signal). Submit on SLURM with the usual machinery plus `--no-eval`
(no corner predictions to score): `python slurm/submit.py both --config
configs/tracker/sam3_candidates.py --jobs 10 --no-eval`.

Explore the dump with `candidate_viewer.py` (FastAPI, like `viewer.py`):
`python candidate_viewer.py pot --config configs/tracker/sam3_candidates.py`.
Overview heatmap (row per sequence, pixel per frame) coloured by candidate
disagreement (`1 − min pairwise mask IoU`) or predicted-IoU spread, sorted
most-ambiguous first; click a sequence for a frame scrubber (arrow keys), an IoU
timeline (predicted + pairwise), and per-candidate togglable overlays. GT is a
secondary togglable overlay plus an optional `regret` layer (where SAM3's
selected candidate is not the one closest to GT). The one-time per-frame spread
precompute runs in parallel (`--workers`) with bbox-cropped IoU (optional
`--stat-downscale`, default full res) and is cached to
`<masks>/candidate_spread_cache.json` (flushed incrementally), so relaunch is
instant.

## Datasets

**POT-210** — `/mnt/datasets/POT-210/`, sequences in `extracted/images/`, GT in `annotation/annotation/`.

**PlanarTrack** — `/mnt/datasets/PlanarTrack/PlanarTrack-Data/`, sequences in `sequences/`, GT in `annos/`, splits in `training-test-splits/`.

**POIC** (20 seq), **TMT** (92 seq), **UCSB** (96 seq) — under `/mnt/personal/serycjon/data/`.
All three share one annotation format, so one loader (`corners_txt`) serves them and the
registry supplies the layout: header line, then `frame ulx uly urx ury lrx lry llx lly`
per frame, corners in upper-left → upper-right → lower-right → lower-left order.

**MPOT-3K** (54 test videos → 573 tasks) — `/mnt/personal/serycjon/data/MPOT-3K/`.
Multi-object, so a tracking task is one planar *instance*, keyed `<video>__<instance>`
and starting at that instance's first annotated frame (the "first" initialisation of the
official protocol). Frames where the instance is absent stay in the frame list and are
marked invalid — MPOT's evaluation charges a false positive for predicting during an
absence, so the tracker must run past it rather than stop.

Two quirks are handled in the loaders and are worth knowing before trusting a number:
TMT's robot sequences pad image and annotation file names to different widths
(`frame00001.jpg` against `frame0001.jpg`), and `robot_bookI` numbers its frames from 0
while numbering its annotations from 1. Annotation rows are therefore matched to frames
**by position** whenever the counts agree — which they do in 189 of the 208
POIC/TMT/UCSB sequences — and only fall back to frame-number matching for the two robot
sequences that ship fewer frames than annotations.

Load with (any dataset key, dispatched through the registry):
```python
from flatsam2.datasets import get_sequences, make_gt_loader, make_image_paths
sequences = get_sequences('ucsb', base_dir)
gt_corners, valid = make_gt_loader('ucsb')(base_dir, seq_name)  # (N,2,4), (N,)
images = make_image_paths('ucsb')(base_dir, seq_name)
```

Check the wiring for the non-POT/PT datasets without a GPU:
```bash
python slurm/submit_cmd.py --gpus 0 -- python experiments/check_new_datasets.py
```

## Legacy baselines (WOFTSAM, SAM-H)

The previous paper's SAM2-based trackers live in a separate, frozen repository
(`~/git/flatsam`). They are not ported; `experiments/run_legacy.py` drives them from
here, so the baselines are produced on exactly the data flatsam2 evaluates on —
including TMT/UCSB (which the legacy loaders do not know) and MPOT-3K's per-instance
tasks.

The split: flatsam2 supplies sequence discovery, GT, init corners and the output layout;
the legacy package supplies the tracker, entered through
`conf.track_function(predictor, conf, frames, init_coords, seq_name, ...)`, which yields
a per-frame homography that is warped into `<seq>.txt` predictions. Config stem names the
results directory (`woftsam_samfallback_nl/`, `WACV2026-SAM-H/`), matching the already
imported POT/PTi runs and `configs/aliases/default.py`.

Environment: the run **must** use the legacy venv — it has `sam2` and `kornia`, which
flatsam2's venv does not, and it satisfies flatsam2's dependencies too. Legacy configs
compose by relative paths and the SAM2.1 checkpoints are `./checkpoints/...`, so the
runner `chdir`s into the legacy repo after resolving every flatsam2-side path.

```bash
# one dataset, 10-way array; the tasks run run_legacy.py under slurm/legacy_env.sh.
# --mem 48G is required: the legacy tracker holds a whole sequence of decoded frames
# in RAM (~3 GB for a POT sequence) on top of SAM2's state, and 16 G is OOM-killed.
python slurm/submit.py tmt --legacy-config ~/git/flatsam/configs/woftsam_samfallback_nl.py --jobs 10 --mem 48G

# scoring a legacy run (its directory is not a flatsam2 config)
python evaluate.py tmt --results-dir "$OUTPUT_DIRECTORY" --results-name woftsam_samfallback_nl
```

## Evaluation

**Alignment error** — RMS across 4 corners: `sqrt(sum((gt - pred)²) / 4)`.

**P@k** — % frames with error ≤ k pixels (k=5,10,15).

**Best rotation** — tries all 4 corner rotations, reports min error (used for both datasets).

```python
from flatsam2.metrics import evaluate_sequence
results = evaluate_sequence(gt, pred, valid, use_best_rotation=True)
# Returns dict with p@5, p@10, p@15, mean_error, br_p@5, etc.
```

## Tensor Reshaping

Use einops (`rearrange`, `reduce`, `repeat`) for all reshaping. Avoid `.reshape`, `.view`, manual transposes.

## Timing / Profiling

`flatsam2/profiling.py` holds a module-level `profiler` singleton. Hot paths in
`tracker.py` and `loma_tracker.py` are wrapped in `with profiler.section(name):`.
When the profiler is disabled (the default) these are near-zero-cost no-ops, so
production tracking is unaffected. When enabled it records hierarchical
wall-clock time per section and (with `sync_cuda=True`) calls
`torch.cuda.synchronize` at section boundaries so GPU work is attributed to the
section that launched it.

Run the harness on one sequence; it writes throwaway corners to a temp dir and
prints a per-building-block report (SAM3, LoMa prep/detect/describe/match/ransac,
Hough mask-H, symmetry, flow) plus overall FPS and peak GPU memory:

```bash
python profile_tracker.py pot --config configs/tracker/loma_samh.py
python profile_tracker.py pot --config configs/tracker/loma_samh.py \
    --sequence V01_1 --max-frames 60 --warmup 5
python profile_tracker.py pot --config configs/tracker/loma_samh.py --no-sync
```

`--warmup` discards the first N tracked frames (cuDNN autotune, lazy model loads)
before measuring; `--no-sync` measures CPU-launch time only. Section percentages
are relative to the per-frame total. Note `slurm/submit.py --pack` packs several
sequences into one SLURM task (GPU sharing), unrelated to per-frame timing — use
this harness to isolate single-stream cost.

## Output

**Predictions** — `{seq}.txt` with (N×8) corner coordinates.

**Evaluation JSON** —
```json
{
  "aggregate": {"mean_p@5": 85.3, "overall_p@5": 87.1, ...},
  "sequences": [{"sequence": "seq_name", "p@5": 90.2, ...}, ...]
}
```
