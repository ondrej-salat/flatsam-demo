"""Visualize detailed real-world evaluation results.

Usage examples:

    python -m flatsam2.nn.visualize_realworld_evaluation \
        /mnt/personal/serycjon/flatsam2/NN_real_data \
        /mnt/personal/serycjon/flatsam2/NN_real_data/eval/nn_quality__best.json

    python -m flatsam2.nn.visualize_realworld_evaluation \
        /mnt/personal/serycjon/flatsam2/NN_real_data \
        /mnt/personal/serycjon/flatsam2/NN_real_data/eval/nn_quality__best__details.npz \
        --top-k 30 \
        --sample-idx 80 1234
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from ..datasets import warp_corners


def _resolve_details_path(result_path: Path) -> Path:
    """Resolve a details NPZ path from either a summary JSON or NPZ."""
    if result_path.suffix == ".npz":
        return result_path

    if result_path.suffix != ".json":
        raise ValueError(f"Unsupported result file: {result_path}")

    with open(result_path, "r") as f:
        result = json.load(f)
    details_path = result.get("details_npz")
    if details_path is None:
        raise ValueError(f"Result JSON does not contain details_npz: {result_path}")
    return Path(details_path)


def _result_stem(result_path: Path) -> str:
    """Create a stable output stem for visualization files."""
    if result_path.name.endswith("__details.npz"):
        return result_path.name[: -len("__details.npz")]
    return result_path.stem


def _load_sequences(dataset_dir: Path) -> Dict[int, dict]:
    """Load sequence metadata keyed by sequence id."""
    sequences_path = dataset_dir / "sequences.json"
    with open(sequences_path, "r") as f:
        sequences = json.load(f)
    return {idx: seq for idx, seq in enumerate(sequences)}


def _render_overlay(
    mask: np.ndarray,
    warped_corners: np.ndarray,
    text_lines: List[str],
) -> np.ndarray:
    """Render one diagnostic image with overlay text and quadrilateral."""
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    polygon = np.rint(warped_corners.T).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(mask.shape[:2]) / 640.0)
    thickness = max(1, int(round(font_scale * 2)))
    line_gap = max(8, int(round(font_scale * 18)))
    padding = max(8, int(round(font_scale * 10)))

    text_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines
    ]
    text_width = max((size[0] for size in text_sizes), default=0)
    text_height = (
        sum(size[1] for size in text_sizes) + max(0, len(text_lines) - 1) * line_gap
    )
    cv2.rectangle(
        vis,
        (0, 0),
        (text_width + 2 * padding, text_height + 2 * padding),
        (0, 0, 0),
        thickness=-1,
    )

    y = padding
    for line, (_, h) in zip(text_lines, text_sizes):
        y += h
        cv2.putText(
            vis,
            line,
            (padding, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap

    return vis


def _load_details(details_path: Path) -> Dict[str, np.ndarray]:
    """Load the saved per-sample evaluation results."""
    with np.load(details_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_sample_visualization(
    dataset_dir: Path,
    sequences: Dict[int, dict],
    details: Dict[str, np.ndarray],
    homographies: np.ndarray,
    sample_pos: int,
    out_path: Path,
) -> None:
    """Write one sample-level debug visualization."""
    seq_id = int(details["seq_id"][sample_pos])
    frame_idx = int(details["frame_idx"][sample_pos])
    seq_name = str(details["seq_name"][sample_pos])
    init_corners = np.asarray(sequences[seq_id]["init_corners"], dtype=np.float64)
    H = homographies[int(details["sample_idx"][sample_pos])]
    mask_path = dataset_dir / "masks" / seq_name / f"{frame_idx + 1:05d}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    warped_corners = warp_corners(H.astype(np.float64), init_corners)
    gt_rmse = float(details["gt_rmse"][sample_pos])
    gt_rmse_raw = float(details.get("gt_rmse_raw", details["gt_rmse"])[sample_pos])
    gt_line = f"gt={gt_rmse:.2f}  pred={float(details['pred_rmse'][sample_pos]):.2f}"
    if abs(gt_rmse_raw - gt_rmse) > 1e-6:
        gt_line = f"gt={gt_rmse:.2f} (raw {gt_rmse_raw:.2f})  pred={float(details['pred_rmse'][sample_pos]):.2f}"
    text_lines = [
        f"sample_idx={int(details['sample_idx'][sample_pos])}",
        f"seq={seq_name} frame={frame_idx + 1:05d}",
        gt_line,
        f"abs_err={float(details['abs_error'][sample_pos]):.2f}  log_sigma={float(details['log_sigma'][sample_pos]):.2f}",
        f"source={str(details['source'][sample_pos])}  config={str(details['config_name'][sample_pos])}",
    ]
    vis = _render_overlay(mask, warped_corners, text_lines)
    ok = cv2.imwrite(str(out_path), vis)
    if not ok:
        raise ValueError(f"Failed to write visualization: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize detailed real-world evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Collected real-world dataset directory.",
    )
    parser.add_argument(
        "result_path",
        type=Path,
        help="Evaluation summary JSON or details NPZ from evaluate_realworld_dataset.py.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Write visualizations for the top-K largest absolute errors.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        nargs="*",
        default=None,
        help="Specific dataset sample indices to visualize.",
    )
    parser.add_argument(
        "--scatter-clip-gt-rmse",
        type=float,
        default=30.0,
        help="Clip GT RMSE values to this maximum for the scatter plot only.",
    )
    args = parser.parse_args()

    details_path = _resolve_details_path(args.result_path)
    details = _load_details(details_path)
    sequences = _load_sequences(args.dataset_dir)
    with np.load(args.dataset_dir / "samples.npz", allow_pickle=False) as samples:
        homographies = samples["homographies"]
    result_stem = _result_stem(details_path)
    out_dir = args.dataset_dir / "eval_vis" / result_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ValueError("matplotlib is required for scatter plot visualization") from e

    gt_rmse = details["gt_rmse"]
    pred_rmse = details["pred_rmse"]
    abs_error = details["abs_error"]
    gt_rmse_scatter = np.clip(gt_rmse, 0.0, args.scatter_clip_gt_rmse)
    max_val = float(max(np.max(gt_rmse_scatter), np.max(pred_rmse), 1.0))

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        gt_rmse_scatter,
        pred_rmse,
        c=abs_error,
        s=10,
        alpha=0.65,
        cmap="viridis",
        linewidths=0,
    )
    ax.plot([0.0, max_val], [0.0, max_val], "r--", linewidth=1)
    ax.set_xlabel("GT RMSE")
    ax.set_ylabel("Predicted RMSE")
    ax.set_title(result_stem)
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Absolute Error")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter.png", dpi=180)
    plt.close(fig)

    requested_positions: List[int] = []
    if args.top_k > 0:
        top_order = np.argsort(-abs_error)[: args.top_k]
        requested_positions.extend(int(pos) for pos in top_order)
    if args.sample_idx:
        sample_to_pos = {
            int(sample_idx): pos for pos, sample_idx in enumerate(details["sample_idx"])
        }
        for sample_idx in args.sample_idx:
            if sample_idx in sample_to_pos:
                requested_positions.append(int(sample_to_pos[sample_idx]))
            else:
                print(f"Sample index {sample_idx} not present in evaluated details")

    unique_positions = []
    seen = set()
    for pos in requested_positions:
        if pos not in seen:
            unique_positions.append(pos)
            seen.add(pos)

    for rank, sample_pos in enumerate(unique_positions, start=1):
        out_path = out_dir / (
            f"{rank:03d}__sample_{int(details['sample_idx'][sample_pos]):06d}"
            f"__err_{float(details['abs_error'][sample_pos]):07.2f}.png"
        )
        _write_sample_visualization(
            args.dataset_dir,
            sequences,
            details,
            homographies,
            sample_pos,
            out_path,
        )

    summary = {
        "details_path": str(details_path),
        "n_points": int(len(gt_rmse)),
        "scatter_clip_gt_rmse": float(args.scatter_clip_gt_rmse),
        "scatter_path": str(out_dir / "scatter.png"),
        "n_sample_visualizations": int(len(unique_positions)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote scatter plot to {out_dir / 'scatter.png'}")
    print(f"Wrote {len(unique_positions)} sample visualizations to {out_dir}")


if __name__ == "__main__":
    main()
