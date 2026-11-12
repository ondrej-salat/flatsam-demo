# FlatSAM: Segmentation-Focused Correspondences for Precise and Robust Planar Tracking

## Abstract
Planar tracking needs precise homographies even when the target is small, blurred, or has weak
texture. Feature matching gives precise geometry, but a limited keypoint budget can be spent on the
background. Segmentation captures target shape, but its corners have ambiguous ordering. We present
FlatSAM, which combines an independently maintained keypoint tracker with a quadrilateral fitted to
a shared SAM3 mask. It prioritises keypoints inside the target, estimates pose directly between the
initial and current frames, and selects between keypoint and mask estimates using match support.
The same matches assign the mask corners, without an additional appearance model. FlatSAM achieves
88.0 p@15 on re-annotated PlanarTrack and 97.23 p@15 on POT-210. It leads on all five benchmarks.
These results show how target-focused matching and mask-based geometry complement each other for
planar tracking.

## Results
| Method | Segmentation model | Geometry branch | POT-210 p@5 | POT-210 p@15 | PlanarTrack p@5 | PlanarTrack p@15 |
|---|---|---|---:|---:|---:|---:|
| SIFT | N/A | keypoints | 65.8 | 69.6 | 18.8 | 26.8 |
| LISRD | N/A | keypoints | 68.3 | 79.2 | 23.5 | 37.3 |
| HDN | N/A | regressed | 70.9 | 92.4 | 31.9 | 58.0 |
| HVC-Net | N/A | regressed | 91.4 | 95.8 | 48.8 | 64.3 |
| WOFT | N/A | flow | 90.0 | 95.4 | 48.9 | 65.9 |
| SAM-H | SAM2 | Hough | 68.6 | 89.4 | 62.4 | 80.5 |
| SAM-H, segmenter control | SAM3 | Hough | 70.9 | 90.3 | <ins>65.0</ins> | <ins>82.4</ins> |
| WOFTSAM | SAM2 | flow+Hough | 91.9 | <ins>97.5</ins> | 57.4 | 77.6 |
| WOFTSAM, segmenter control | SAM3 | flow+Hough | 92.1 | 97.2 | 58.3 | 79.2 |
| keypoint branch only (ours) | SAM3 | keypoints | <ins>92.4</ins> | 97.3 | 56.7 | 78.2 |
| **FlatSAM (ours)** | SAM3 | keypoints+Hough | **93.4** | **98.5** | **65.6** | **88.0** |

Pooled p@k (%) on the 1,004 changed-frame POT-210 protocol and re-annotated PlanarTrack. POT-210
values follow WOFTSAM; PlanarTrack uses WOFTSAM's updated initial poses. SIFT, LISRD, and HDN are
rescored released predictions (not reruns), with fallback on 236 degenerate frames. POT-210 uses
WOFT's changed-frame labels; our PlanarTrack rows are evaluated predictions. SAM3 controls change
only the segmenter.

## Installation
```bash
cd flatsam-demo

./setup.sh

source .envrc
```

`setup.sh` needs a Hugging Face token with access to the gated [SAM3 weights](https://huggingface.co/facebook/sam3).
Request access there, create a Read token at <https://huggingface.co/settings/tokens>, and paste it
when `setup.sh` asks (or set `HF_TOKEN` in `.env`). Settings (dataset and output paths) live in
`.env`, created from `.env.example` on the first run.

Tested on Linux x86_64 with Python 3.12 and CUDA 12.6, on NVIDIA A100 and RTX 4000 SFF Ada (20 GB) GPUs.
