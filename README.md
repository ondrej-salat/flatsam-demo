# FlatSAM2

Planar object tracker combining SAM segmentation with optical flow to track surfaces across video frames.

## Features

- Track planar surfaces using optical flow and SAM
- POT-210 and PlanarTrack dataset support
- Alignment error, P@5/10/15 metrics with best-rotation handling

## Demo (any Linux machine with an NVIDIA GPU)

Needs: Linux x86_64, an NVIDIA GPU with ≥ 8 GB and a driver for CUDA 12.6 (≥ 560; not RTX 50xx),
git, curl or wget, internet, ~20 GB disk. No Python needed: `setup.sh` fetches Python 3.12 via
[uv](https://docs.astral.sh/uv/) and installs the pinned `requirements-lock.txt` into `venv/`.

SAM3 is gated: request access at <https://huggingface.co/facebook/sam3> and create a Read token at
<https://huggingface.co/settings/tokens>. `setup.sh` asks for it once (or set `HF_TOKEN` in `.env`).

```bash
./setup.sh && source .envrc       # once; in every new shell: source .envrc
python demo.py video clip.mp4 --corners x1 y1 x2 y2 x3 y3 x4 y4  # object corners in frame 1
python demo.py dataset pot V01_7  # needs POT-210 in data/POT-210/ (paths in .env)
python demo.py params             # tunable keys; override any with --set key=value
```

`video` writes `clip_tracked.mp4` next to the input; `dataset` writes predictions and metrics to
`results/` and a video to `vis/`. A 500-frame sequence takes ~5 min (RTX 4000 Ada).

On the RCI cluster: same steps on the login node, with `HF_HOME`, `TORCH_HOME` and the dataset paths
in `.env` pointing to `/mnt`; add `--slurm` to run the demo on a GPU node.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Run tracking on POT-210 with default config
# Creates: ./results/default/POT-210/
flatsam2 run pot --output-dir ./results

# Run on PlanarTrack test split with custom config
# Creates: ./results/fast_kitti/PlanarTrack/
flatsam2 run planartrack --split test --config configs/tracker/fast_kitti.py --output-dir ./results

# Evaluate results (config must match the one used for tracking)
flatsam2 evaluate pot --results-dir ./results --config configs/tracker/fast_kitti.py --output eval.json

# Visualize a sequence
flatsam2 visualize pot ./results sequence_name --config configs/tracker/fast_kitti.py

# Build standalone HTML table report from collected eval results
python collect_results.py \
  --table-report-out results_table_report.html \
  --views-config configs/reports/example_views.py
```

Override dataset paths with `--base-dir` (default: `/mnt/datasets/POT-210/` or `/mnt/datasets/PlanarTrack/PlanarTrack-Data/`).

Run specific sequences or save masks with `--sequences` / `--save-masks`.

## Standalone Results Table Report

`collect_results.py` can produce fully offline HTML report with named views (slides)
plus live in-browser edits (per-row show/hide toggles, column toggle, row drag reorder,
row renaming, baseline diff mode with colored deltas, and ViewSpec export/copy).
Live edits are intentionally non-persistent.

```bash
python collect_results.py \
  --datasets pot pti poic tmt ucsb mpot \
  --table-report-out results_table_report.html \
  --views-config configs/reports/example_views.py
```

`--datasets` picks which dataset column groups appear (registry keys, in order;
default `pot pti`). The story views in `configs/reports/example_views.py` expect
all six and show empty columns without them.

View spec is Python file defining `report` variable of type `ReportSpec`
(see `configs/reports/example_views.py`).

## PlanarTrack Sequence Attribute Annotator

Fast sequence-level labeling UI for PlanarTrack attributes (binary toggles).

```bash
python planartrack_sequence_annotator.py \
  --split test \
  --attributes appearance-changing box-shaped box-shaped-flat-init
```

Key controls:
- `ArrowLeft` / `ArrowRight` → previous / next sequence
- `Numpad1..Numpad9` → toggle attributes by list order
- `Ctrl+Wheel` or zoom slider → zoom into frame tiles

Labels auto-save into `./annotations/planartrack_sequence_attributes.json` (override via
`--annotations-file`) and auto-load on restart.

## Directory Structure

FlatSAM2 organizes outputs hierarchically: `output_dir / config_name / dataset_name /`

```
results/
├── default/                 # Results using default config
│   ├── POT-210/
│   │   ├── sequence1.txt
│   │   └── sequence2.txt
│   └── PlanarTrack/
│       ├── sequence1.txt
│       └── sequence2.txt
└── fast_kitti/              # Results using fast_kitti config
    ├── POT-210/
    │   └── ...
    └── PlanarTrack/
        └── ...
```

The `--output-dir` argument specifies only the **top-level** directory. Subdirectories are created automatically based on:
- Config file name (or `default` if no config specified)
- Dataset name (`POT-210` or `PlanarTrack`)

This structure applies to:
- `flatsam2 run` — creates output directories
- `flatsam2 evaluate` — reads from structured directories
- `flatsam2 visualize` — reads from and writes to structured directories

## Configuration

FlatSAM uses Python-based configs with [pydantic](https://docs.pydantic.dev/) for modularity and type safety.

### Using Configs

```bash
# Use a config file
flatsam2 run pot --config configs/tracker/default.py --output-dir ./results
```

### Creating Configs

```python
# configs/my_experiment.py
from flatsam2.config import TrackerConfig, FlowConfig, SegmenterConfig

config = TrackerConfig(
    flow=FlowConfig(
        model="raft",           # ptlflow model: 'raft', 'gma', 'unimatch', etc.
        weights="kitti",        # weights variant: 'sintel', 'kitti', 'things'
        ransac_threshold=3.0,   # RANSAC threshold in pixels
        use_cuda=True
    ),
    segmenter=SegmenterConfig(
        model="sam3"
    )
)
```

### Composing Configs

Reuse components across experiments:

```python
# configs/optical_flow/raft_kitti.py
from flatsam2.config import FlowConfig
config = FlowConfig(model="raft", weights="kitti")

# configs/tracker/experiment_v2.py
from configs.optical_flow.raft_kitti import config as flow_config
from flatsam2.config import TrackerConfig, SegmenterConfig

config = TrackerConfig(
    flow=flow_config,
    segmenter=SegmenterConfig(model="sam3")
)
```

See `configs/` directory for examples.
## Python API

### Tracking

```python
import cv2
import numpy as np
from flatsam2.tracker import FlatSAMTracker
from flatsam2.config import TrackerConfig, FlowConfig
from flatsam2.datasets import warp_corners

# Create tracker with optional config
config = TrackerConfig(
    flow=FlowConfig(model="raft", weights="sintel")
)
tracker = FlatSAMTracker(config)

# Initialize with corners (planar benchmarks)
frame = cv2.imread('frame_0001.jpg')
init_corners = np.array([[x1, x2, x3, x4],
                         [y1, y2, y3, y4]])  # (2, 4)
tracker.init(frame, init_corners=init_corners)

# OR initialize with mask (general shapes)
# tracker.init(frame, init_mask=binary_mask)

# Track through frames
for frame_path in frame_paths[1:]:
    frame = cv2.imread(frame_path)
    H = tracker.track(frame)  # (3, 3) homography
    current_corners = warp_corners(H, init_corners)
```

### Dataset Loading

```python
from pathlib import Path
from flatsam2.datasets import pot_sequences, pot_load_gt, pot_image_paths

base_dir = Path('/mnt/datasets/POT-210')
for seq in pot_sequences(base_dir):
    gt_corners, valid = pot_load_gt(base_dir, seq)  # (N,2,4), (N,)
    images = pot_image_paths(base_dir, seq)
```

### Evaluation

```python
from flatsam2.metrics import evaluate_sequence

results = evaluate_sequence(gt, pred, valid, use_best_rotation=True)
print(f"P@5: {results['p@5']:.1f}%")
```

## Metrics

- **Alignment error** — RMS distance across 4 corners
- **P@5/10/15** — % frames with error ≤ 5/10/15 pixels
- **Best rotation** — rotation-invariant evaluation (all 4 corner orderings)

## License

MIT
