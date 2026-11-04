# FlatSAM2 demo

Planar object tracker: give it a video and the 4 corners of a planar object (poster, book cover,
screen, ...) in the first frame, and it tracks the object's homography through the video. It
combines LoMa keypoint matching with SAM3 segmentation: a keypoint homography while there are
enough matches, a homography from the SAM3 mask when there are not.

## Requirements

- Linux x86_64, an NVIDIA GPU with ≥ 8 GB and a driver for CUDA 12.6 (≥ 560; not RTX 50xx)
- git, curl or wget, internet, ~20 GB disk
- No Python needed: `setup.sh` fetches Python 3.12 via [uv](https://docs.astral.sh/uv/) and installs
  the pinned `requirements-lock.txt` into `venv/`
- A Hugging Face account with access to the gated SAM3 weights: request it at
  <https://huggingface.co/facebook/sam3>, then create a Read token at
  <https://huggingface.co/settings/tokens>

## Setup

```bash
./setup.sh && source .envrc       # once; asks for the HF token (or set HF_TOKEN in .env)
```

In every new shell: `source .envrc`. Settings (output and dataset paths) live in `.env`, created
from `.env.example` on the first run.

## Run

```bash
python demo.py video clip.mp4 --corners x1 y1 x2 y2 x3 y3 x4 y4
python demo.py dataset pot V01_7
python demo.py params             # tunable keys; override any with --set key=value
```

- `video`: `--corners` are the object's corners in the first frame, in pixels, clockwise from
  top-left. Writes `clip_tracked.mp4` next to the input.
- `dataset`: tracks one POT-210 sequence, evaluates it and renders it. Expects
  `data/POT-210/extracted/images/<seq>/*.jpg` and
  `data/POT-210/annotation/WOFT_reannotation_coarse_onlychanged/<seq>_{gt_points,flag}.txt`
  (paths in `.env`). Writes predictions and metrics to `results/` and a video to `vis/`.

A 500-frame 1280×720 sequence takes ~3–5 min (A100 / RTX 4000 Ada).

## Python API

```python
import numpy as np
from flatsam2.config import load_config_from_file
from flatsam2.datasets import warp_corners
from flatsam2.tracker import FlatSAMTracker

tracker = FlatSAMTracker(load_config_from_file("configs/tracker/loma_samh_matchnaming.py"))
corners = np.array([[x1, x2, x3, x4],
                    [y1, y2, y3, y4]], dtype=float)  # (2, 4), first frame
tracker.init(first_frame, init_corners=corners)      # BGR uint8 (H, W, 3)
for frame in frames:
    H = tracker.track(frame).homography              # (3, 3), first frame -> this frame
    current = warp_corners(H, corners)               # (2, 4)
```

## Metrics (`dataset` mode)

- **Alignment error**: RMS distance over the 4 corners
- **P@5 / P@15**: % of frames with alignment error ≤ 5 / 15 px

## License

MIT
