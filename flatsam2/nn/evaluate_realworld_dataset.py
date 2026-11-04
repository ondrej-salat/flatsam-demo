"""Evaluate NN homography-quality checkpoints on a collected real-world dataset.

Usage examples:

    python -m flatsam2.nn.evaluate_realworld_dataset \
        /scratch/datasets/realworld_val \
        checkpoints/boundary/best.pt \
        checkpoints/point_baseline/best.pt

    python -m flatsam2.nn.evaluate_realworld_dataset \
        /scratch/datasets/realworld_val \
        checkpoints/boundary_run \
        --subset-size 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from .collect_realworld_dataset import _DEFAULT_BIN_EDGES, _format_bin_label
from .data_gen import RealWorldHomographyDataset, collate_skip_none
from .inference import load_model_checkpoint
from .train import compute_val_metrics, forward_model_batch


def _resolve_checkpoint_path(path: Path) -> Path:
    """Resolve a checkpoint file path from a file or run directory."""
    if path.is_file():
        return path

    if not path.is_dir():
        raise ValueError(f"Checkpoint path does not exist: {path}")

    for candidate_name in ("best.pt", "final.pt"):
        candidate = path / candidate_name
        if candidate.exists():
            return candidate

    raise ValueError(f"No checkpoint file found in directory: {path}")


def _sanitize_checkpoint_name(path: Path) -> str:
    """Convert a checkpoint path into a filename-safe stem."""
    parent = path.parent.name or "checkpoint"
    return f"{parent}__{path.stem}"


def _compute_uncensored_nll(
    pred_rmse: np.ndarray,
    log_sigma: np.ndarray,
    gt_rmse: np.ndarray,
) -> float:
    """Compute the uncensored heteroscedastic regression objective."""
    log_sigma = np.clip(log_sigma, -5.0, 5.0)
    return float(
        np.mean((pred_rmse - gt_rmse) ** 2 * np.exp(-2.0 * log_sigma) + 2.0 * log_sigma)
    )


def _clip_gt_rmse_for_eval(gt_rmse: np.ndarray, clip_value: float) -> np.ndarray:
    """Clip GT RMSE for regression-style evaluation metrics."""
    return np.clip(gt_rmse, 0.0, clip_value)


def _compute_bin_metrics(
    pred_rmse: np.ndarray,
    gt_rmse: np.ndarray,
    bin_edges: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Summarize prediction quality per GT RMSE bin."""
    bin_ids = np.digitize(gt_rmse, bin_edges) - 1
    metrics: Dict[str, Dict[str, float]] = {}
    for bin_idx in range(len(bin_edges) - 1):
        in_bin = bin_ids == bin_idx
        if not np.any(in_bin):
            continue

        label = _format_bin_label(bin_edges[bin_idx], bin_edges[bin_idx + 1])
        pred_bin = pred_rmse[in_bin]
        gt_bin = gt_rmse[in_bin]
        metrics[label] = {
            "n_samples": int(np.sum(in_bin)),
            "gt_mean": float(np.mean(gt_bin)),
            "pred_mean": float(np.mean(pred_bin)),
            "mae": float(np.mean(np.abs(pred_bin - gt_bin))),
            "rmse": float(np.sqrt(np.mean((pred_bin - gt_bin) ** 2))),
        }
    return metrics


def evaluate_checkpoint(
    dataset_dir: Path,
    checkpoint_path: Path,
    subset_size: Optional[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dir: Path,
    gt_rmse_clip: float,
) -> dict:
    """Evaluate one checkpoint on the real-world dataset."""
    model, state, arch = load_model_checkpoint(checkpoint_path, device)
    checkpoint_config = state.get("config", {})
    model.to(device)
    model.eval()

    dataset = RealWorldHomographyDataset(
        str(dataset_dir),
        architecture=arch,
        subset_size=subset_size,
        n_points=int(checkpoint_config.get("n_points", 128)),
        sdf_downsample=int(checkpoint_config.get("sdf_downsample", 4)),
        crop_size=int(checkpoint_config.get("crop_size", 128)),
        augment_masks=False,
        return_info=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"No samples available for evaluation in {dataset_dir}")
    try:
        available_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        available_workers = os.cpu_count() or 1
    eval_num_workers = max(0, min(num_workers, available_workers))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_skip_none,
    )

    all_pred_rmse: List[np.ndarray] = []
    all_log_sigma: List[np.ndarray] = []
    all_gt_rmse: List[np.ndarray] = []
    all_sample_idx: List[np.ndarray] = []
    all_seq_id: List[np.ndarray] = []
    all_frame_idx: List[np.ndarray] = []
    all_seq_names: List[np.ndarray] = []
    all_sources: List[np.ndarray] = []
    all_config_names: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            if batch is None:
                continue
            batch_info = batch[-1]
            batch_inputs = batch[:-1]
            pred_rmse, log_sigma, gt_rmse, _ = forward_model_batch(
                model,
                arch,
                batch_inputs,
                device,
            )
            all_pred_rmse.append(pred_rmse.cpu().numpy())
            all_log_sigma.append(log_sigma.cpu().numpy())
            all_gt_rmse.append(gt_rmse.cpu().numpy())
            all_sample_idx.append(batch_info["sample_idx"].cpu().numpy())
            all_seq_id.append(batch_info["seq_id"].cpu().numpy())
            all_frame_idx.append(batch_info["frame_idx"].cpu().numpy())
            all_seq_names.append(np.asarray(batch_info["seq_name"]))
            all_sources.append(np.asarray(batch_info["source"]))
            all_config_names.append(np.asarray(batch_info["config_name"]))

    if not all_pred_rmse:
        raise ValueError(f"No valid evaluation samples produced for {checkpoint_path}")

    pred = np.concatenate(all_pred_rmse)
    log_sigma = np.concatenate(all_log_sigma)
    gt = np.concatenate(all_gt_rmse)
    gt_clipped = _clip_gt_rmse_for_eval(gt, gt_rmse_clip)
    sample_idx = np.concatenate(all_sample_idx)
    seq_id = np.concatenate(all_seq_id)
    frame_idx = np.concatenate(all_frame_idx)
    seq_names = np.concatenate(all_seq_names)
    sources = np.concatenate(all_sources)
    config_names = np.concatenate(all_config_names)
    abs_error = np.abs(pred - gt_clipped)

    metrics = compute_val_metrics(pred, gt)
    metrics.update(
        {
            "mae": float(np.mean(np.abs(pred - gt_clipped))),
            "rmse": float(np.sqrt(np.mean((pred - gt_clipped) ** 2))),
            "mean_pred_rmse": float(np.mean(pred)),
            "mean_gt_rmse": float(np.mean(gt_clipped)),
            "mean_gt_rmse_raw": float(np.mean(gt)),
            "gt_rmse_clip": float(gt_rmse_clip),
            "uncensored_nll": _compute_uncensored_nll(pred, log_sigma, gt_clipped),
        }
    )

    bin_edges = _DEFAULT_BIN_EDGES.astype(np.float64)
    meta_path = dataset_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "collection_bin_edges" in meta:
            bin_edges = np.asarray(meta["collection_bin_edges"], dtype=np.float64)

    detail_stem = _sanitize_checkpoint_name(checkpoint_path)
    if subset_size is not None:
        detail_stem = f"{detail_stem}__subset{subset_size}"
    details_npz_path = output_dir / f"{detail_stem}__details.npz"
    details_csv_path = output_dir / f"{detail_stem}__details.csv"
    np.savez(
        details_npz_path,
        sample_idx=sample_idx,
        seq_id=seq_id,
        frame_idx=frame_idx,
        seq_name=seq_names,
        source=sources,
        config_name=config_names,
        gt_rmse=gt_clipped,
        gt_rmse_raw=gt,
        pred_rmse=pred,
        log_sigma=log_sigma,
        abs_error=abs_error,
    )
    with open(details_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_idx",
                "seq_id",
                "frame_idx",
                "seq_name",
                "source",
                "config_name",
                "gt_rmse",
                "gt_rmse_raw",
                "pred_rmse",
                "log_sigma",
                "abs_error",
            ]
        )
        for row_idx in range(len(pred)):
            writer.writerow(
                [
                    int(sample_idx[row_idx]),
                    int(seq_id[row_idx]),
                    int(frame_idx[row_idx]),
                    str(seq_names[row_idx]),
                    str(sources[row_idx]),
                    str(config_names[row_idx]),
                    float(gt_clipped[row_idx]),
                    float(gt[row_idx]),
                    float(pred[row_idx]),
                    float(log_sigma[row_idx]),
                    float(abs_error[row_idx]),
                ]
            )

    return {
        "checkpoint_path": str(checkpoint_path),
        "arch": arch,
        "subset_size": subset_size,
        "n_samples": int(len(gt)),
        "details_npz": str(details_npz_path),
        "details_csv": str(details_csv_path),
        "gt_rmse_clip": float(gt_rmse_clip),
        "checkpoint_metrics": state.get("metrics", {}),
        "checkpoint_config": checkpoint_config,
        "metrics": metrics,
        "bin_metrics": _compute_bin_metrics(pred, gt_clipped, bin_edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints on a collected real-world dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory produced by collect_realworld_dataset.py.",
    )
    parser.add_argument(
        "checkpoints",
        type=Path,
        nargs="+",
        help="Checkpoint files or run directories containing best.pt/final.pt.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional subset size from dataset/subsets/subset_<N>.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data-loading workers.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON summaries (default: <dataset_dir>/eval).",
    )
    parser.add_argument(
        "--gt-rmse-clip",
        type=float,
        default=30.0,
        help="Clip GT RMSE to this maximum for regression-style evaluation metrics.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.dataset_dir / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    for checkpoint_input in args.checkpoints:
        checkpoint_path = _resolve_checkpoint_path(checkpoint_input)
        result = evaluate_checkpoint(
            args.dataset_dir,
            checkpoint_path,
            args.subset_size,
            args.batch_size,
            args.num_workers,
            device,
            output_dir,
            args.gt_rmse_clip,
        )

        out_stem = _sanitize_checkpoint_name(checkpoint_path)
        if args.subset_size is not None:
            out_stem = f"{out_stem}__subset{args.subset_size}"
        out_name = out_stem + ".json"
        out_path = output_dir / out_name
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        metrics = result["metrics"]
        print(
            f"{checkpoint_path}: arch={result['arch']}  n={result['n_samples']}  "
            f"mae={metrics['mae']:.3f}  rmse={metrics['rmse']:.3f}  "
            f"acc@5={metrics['accuracy_5px']:.3f}  "
            f"acc@15={metrics['accuracy_15px']:.3f}  "
            f"bal@5={metrics['balanced_acc_5px']:.3f}  "
            f"bal@15={metrics['balanced_acc_15px']:.3f}  "
            f"rho@15={metrics['spearman_rho_15px']:.3f}  "
            f"gt_clip={result['gt_rmse_clip']:.1f}"
        )
        print(f"  saved: {out_path}")
        print(f"  details: {result['details_npz']}")
        print(f"  csv: {result['details_csv']}")


if __name__ == "__main__":
    main()
