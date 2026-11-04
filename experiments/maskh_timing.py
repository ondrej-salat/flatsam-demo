#!/usr/bin/env python3
"""Publishable timing for the mask -> homography estimators.

``maskh_bench`` reports a millisecond column, but it runs 16 worker processes on
one node, so those numbers are contended and only their ratios mean anything.
This measures the same estimators the way a paper has to quote them: one
process, one thread, warm-up discarded, repeated trials, median of per-frame
times — plus the internal stage breakdown from ``flatsam2.profiling``.

Two numbers matter and they are different:

* **fast path** — the incumbent pose verifies, the search is skipped. This is
  what ~88% of tracked frames actually cost.
* **search path** — the incumbent is rejected and the stateless sweep runs.
  This is the worst case, and the honest number to quote next to a redetector.

The benchmark's ``good`` regime exercises the first, ``cold`` the second, so the
same case list measured under both regimes gives both columns.

Threads are pinned to one core (OpenCV and BLAS both) so the result does not
silently depend on how many cores the node happened to give us.

Imports cv2 — submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 --cpus 4 -- \\
        python experiments/maskh_timing.py --cases 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np

cv2.setNumThreads(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.path_config import paths
from flatsam2.profiling import profiler
from flatsam2.registry import DATASETS

_BENCH_DIR = paths.OUTPUT_DIRECTORY / "experiments" / "maskh_bench"


def _load_cases(dataset: str, split: str, limit: int, manifest: str) -> List[Dict]:
    """Benchmark cases with their masks, frames and poses already resolved."""
    from flatsam2.datasets import pot_image_paths, pt_image_paths

    cases = [
        c for c in json.loads((_BENCH_DIR / manifest).read_text())["cases"]
        if c["dataset"] == dataset and c["split"] == split
    ]
    label = DATASETS[dataset].label
    base_dir = getattr(paths, DATASETS[dataset].base_dir_attr)
    image_paths_of = pot_image_paths if DATASETS[dataset].gt_loader == "pot" else pt_image_paths

    loaded, by_seq = [], {}
    for case in cases:
        if len(loaded) >= limit:
            break
        seq = case["seq"]
        if seq not in by_seq:
            mask_dir = paths.OUTPUT_DIRECTORY / "sam3_candidates" / label / "masks" / seq
            template = cv2.imread(str(mask_dir / "00001.png"), cv2.IMREAD_GRAYSCALE)
            try:
                frames = image_paths_of(base_dir, seq)
            except ValueError:
                frames = None
            by_seq[seq] = (mask_dir, template, frames)
        mask_dir, template, frames = by_seq[seq]
        if template is None or frames is None:
            continue
        mask = cv2.imread(str(mask_dir / f"{case['frame_idx'] + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or case["frame_idx"] >= len(frames):
            continue
        loaded.append({
            "mask": mask > 127,
            "template_mask": (template > 127).astype(np.uint8) * 255,
            "corners": np.array(case["template_corners"]),
            "inits": {k: np.array(v) for k, v in case["inits"].items()},
            "init_frame": cv2.imread(str(frames[0]), cv2.IMREAD_COLOR),
            "frame": cv2.imread(str(frames[case["frame_idx"]]), cv2.IMREAD_COLOR),
            "shape": mask.shape,
        })
    return loaded


def _H_of(corners: np.ndarray, target: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(
        corners.T.astype(np.float32), target.T.astype(np.float32)
    )


def _time_estimator(cases: List[Dict], regime: str, fn, warmup: int, repeats: int) -> Dict:
    """Median and spread of per-frame time for one estimator under one regime."""
    for case in cases[:warmup]:
        fn(case, regime)
    per_frame = []
    for _ in range(repeats):
        for case in cases:
            t0 = time.perf_counter()
            fn(case, regime)
            per_frame.append(1000.0 * (time.perf_counter() - t0))
    arr = np.array(per_frame)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
        "n": len(arr),
    }


def _estimators(register_cfg_overrides: Optional[Dict] = None) -> Dict[str, object]:
    """The estimators to time, each as ``(case, regime) -> None``."""
    from flatsam2.hough_homography import mask_to_homography
    from flatsam2.mask_align import align_mask_to_homography
    from flatsam2.mask_register import RegisterConfig, RegistrationTemplate, register_mask

    cfg = RegisterConfig(**(register_cfg_overrides or {}))
    templates: Dict[int, object] = {}

    def _template(case: Dict):
        """Per-sequence context, built once exactly as the tracker builds it."""
        key = id(case["template_mask"])
        if key not in templates:
            templates[key] = RegistrationTemplate.build(
                case["corners"], case["template_mask"], case["shape"], cfg,
                init_frame=case["init_frame"],
            )
        return templates[key]

    def hough(case, regime):
        mask_to_homography(
            case["mask"].astype(np.uint8) * 255, case["corners"],
            _H_of(case["corners"], case["inits"][regime]),
        )

    def align(case, regime):
        align_mask_to_homography(
            case["mask"], case["corners"], _H_of(case["corners"], case["inits"][regime]),
            template_mask=case["template_mask"],
        )

    def register(case, regime):
        template = _template(case)
        if template is None:
            return
        register_mask(
            template, case["mask"], last_H=_H_of(case["corners"], case["inits"][regime]),
            cfg=cfg, frame=case["frame"],
        )

    def register_kp(case, regime):
        template = _template(case)
        if template is None:
            return
        # A keypoint proposal at the true pose is the favourable case for the
        # fast path; timing it separately shows what the extra solve costs.
        register_mask(
            template, case["mask"], last_H=_H_of(case["corners"], case["inits"][regime]),
            extra_H=(_H_of(case["corners"], case["inits"]["good"]),), cfg=cfg,
            frame=case["frame"],
        )

    return {"hough": hough, "align": align, "register": register, "register_kp": register_kp}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["pti", "pot"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--cases", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--regimes", nargs="+", default=["good", "cold"])
    parser.add_argument("--manifest", default="testset_v1.json")
    parser.add_argument("--stages", action="store_true",
                        help="also report the registrar's internal stage breakdown")
    args = parser.parse_args(argv)

    print(f"single-process timing | cv2 threads={cv2.getNumThreads()} | "
          f"cases<={args.cases} warmup={args.warmup} repeats={args.repeats}\n")

    rows = []
    for dataset in args.datasets:
        cases = _load_cases(dataset, args.split, args.cases, args.manifest)
        if not cases:
            print(f"{dataset}: no cases")
            continue
        h, w = cases[0]["shape"]
        print(f"=== {DATASETS[dataset].label}  n={len(cases)}  frame={w}x{h}")
        print(f"  {'estimator':<14}" + "".join(f"{r + ' med/p90':>22}" for r in args.regimes))
        estimators = _estimators()
        for name, fn in estimators.items():
            cells = ""
            for regime in args.regimes:
                stat = _time_estimator(cases, regime, fn, args.warmup, args.repeats)
                rows.append({"dataset": dataset, "estimator": name, "regime": regime, **stat})
                cells += f"{stat['median_ms']:14.1f} /{stat['p90_ms']:6.1f}"
            print(f"  {name:<14}{cells}")
        print()

        if args.stages:
            for regime in args.regimes:
                profiler.reset()
                profiler.enable(sync_cuda=False)
                for case in cases:
                    estimators["register"](case, regime)
                profiler.disable()
                print(f"--- register stage breakdown, {dataset} / {regime} "
                      f"({len(cases)} frames)")
                print(profiler.report(n_frames=len(cases)))

    out = _BENCH_DIR / "timing.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
