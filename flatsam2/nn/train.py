"""Training script for NN homography quality estimation.

Usage:
    python -m flatsam2.nn.train --arch option_e --output-dir checkpoints/option_e
    python -m flatsam2.nn.train --arch option_c --output-dir checkpoints/option_c
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .collect_realworld_dataset import _DEFAULT_BIN_EDGES, create_stratified_subset
from .data_gen import (
    CachedHomographyDataset,
    RealWorldHomographyDataset,
    SyntheticHomographyDataset,
    collate_skip_none,
)
from .loss import censored_heteroscedastic_loss
from .models import BoundaryTransformer, PointSetBaseline, SDFCropCNN


@dataclass
class TrainConfig:
    arch: str = "option_e"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_points: int = 128
    crop_size: int = 128

    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    grad_clip: float = 1.0
    n_clip: float = 30.0

    epochs: int = 50
    steps_per_epoch: int = 10_000
    val_steps: int = 5_000
    patience: int = 5

    num_workers: int = 4
    seed: int = 42
    output_dir: str = "checkpoints/nn_quality"
    sdf_downsample: int = 4
    cache_dir: Optional[str] = None
    realworld_dataset_dir: Optional[str] = None
    realworld_train_subset_size: Optional[int] = None
    realworld_val_subset_size: Optional[int] = None
    realworld_train_augment: bool = False


def compute_val_metrics(
    pred_rmse: np.ndarray,
    gt_rmse: np.ndarray,
    thresholds: tuple[float, ...] = (5.0, 15.0),
) -> Dict[str, float]:
    """Compute thresholded classification metrics and ranking correlation."""
    metrics: Dict[str, float] = {}

    for threshold in thresholds:
        label = f"{int(threshold)}px"
        gt_good = gt_rmse < threshold
        gt_bad = ~gt_good
        pred_good = pred_rmse < threshold

        tpr = float((pred_good & gt_good).sum() / max(gt_good.sum(), 1))
        tnr = float((~pred_good & gt_bad).sum() / max(gt_bad.sum(), 1))
        balanced_acc = (tpr + tnr) / 2
        accuracy = float(np.mean(pred_good == gt_good))

        metrics[f"accuracy_{label}"] = accuracy
        metrics[f"tpr_{label}"] = tpr
        metrics[f"tnr_{label}"] = tnr
        metrics[f"balanced_acc_{label}"] = balanced_acc

    rho = float("nan")
    gt_good_15 = gt_rmse < 15.0
    if gt_good_15.sum() > 10:
        from scipy.stats import spearmanr

        rho_val, _ = spearmanr(-pred_rmse[gt_good_15], -gt_rmse[gt_good_15])
        rho = float(rho_val)
    metrics["spearman_rho_15px"] = rho

    return metrics


def forward_model_batch(
    model: torch.nn.Module,
    arch: str,
    batch,
    device: torch.device,
):
    """Run one batch through the selected architecture."""
    if arch == "option_e":
        feat_A, feat_B, gt_rmse, censored = [b.to(device) for b in batch]
        pred_rmse, log_sigma = model(feat_A, feat_B)
        return pred_rmse, log_sigma, gt_rmse, censored

    if arch == "point_baseline":
        features, gt_rmse, censored = [b.to(device) for b in batch]
        pred_rmse, log_sigma = model(features)
        return pred_rmse, log_sigma, gt_rmse, censored

    crop, scale, gt_rmse, censored = [b.to(device) for b in batch]
    pred_rmse_crop, log_sigma_crop = model(crop)
    pred_rmse = pred_rmse_crop / scale
    log_sigma = log_sigma_crop - torch.log(scale)
    return pred_rmse, log_sigma, gt_rmse, censored


def build_datasets(config: TrainConfig):
    """Construct train and validation datasets from the configured sources."""
    if (
        config.realworld_train_subset_size is not None
        and config.realworld_dataset_dir is None
    ):
        raise ValueError(
            "realworld_dataset_dir must be set when using real-world training"
        )
    realworld_train_indices = None
    realworld_val_indices = None
    if config.realworld_train_subset_size is not None:
        if config.realworld_val_subset_size is None:
            raise ValueError(
                "realworld_val_subset_size must be set when using real-world training"
            )
        realworld_train_indices, realworld_val_indices = build_realworld_split_indices(
            config,
        )

    if config.realworld_dataset_dir is not None:
        val_ds = RealWorldHomographyDataset(
            config.realworld_dataset_dir,
            architecture=config.arch,
            subset_size=(
                config.realworld_val_subset_size
                if realworld_val_indices is None
                else None
            ),
            sample_indices=realworld_val_indices,
            n_points=config.n_points,
            sdf_downsample=config.sdf_downsample,
            crop_size=config.crop_size,
            augment_masks=False,
        )
        val_source = "real-world"
    elif config.cache_dir is not None:
        cache_arch = config.arch
        val_ds = CachedHomographyDataset(config.cache_dir, cache_arch, augment=False)
        val_source = "cache"
    else:
        val_ds = SyntheticHomographyDataset(
            architecture=config.arch,
            seed=999999,
            n_points=config.n_points,
            max_samples=config.val_steps,
            sdf_downsample=config.sdf_downsample,
            crop_size=config.crop_size,
        )
        val_source = "synthetic"

    if config.realworld_train_subset_size is not None:
        train_ds = RealWorldHomographyDataset(
            config.realworld_dataset_dir,
            architecture=config.arch,
            sample_indices=realworld_train_indices,
            n_points=config.n_points,
            sdf_downsample=config.sdf_downsample,
            crop_size=config.crop_size,
            augment_masks=config.realworld_train_augment,
        )
        train_source = "real-world"
    elif config.cache_dir is not None:
        cache_arch = config.arch
        train_ds = CachedHomographyDataset(config.cache_dir, cache_arch, augment=True)
        train_source = "cache"
    else:
        train_ds = SyntheticHomographyDataset(
            architecture=config.arch,
            seed=config.seed,
            n_points=config.n_points,
            sdf_downsample=config.sdf_downsample,
            crop_size=config.crop_size,
        )
        train_source = "synthetic"

    return train_ds, val_ds, train_source, val_source


def resolve_num_workers(requested: int) -> int:
    """Clamp worker count to the visible CPU budget."""
    if requested <= 0:
        return 0
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(0, min(requested, available))


def build_realworld_split_indices(
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Create disjoint train/validation splits from the real-world sample pool."""
    if config.realworld_dataset_dir is None:
        raise ValueError("realworld_dataset_dir must be set for real-world splits")
    if (
        config.realworld_train_subset_size is None
        or config.realworld_val_subset_size is None
    ):
        raise ValueError("Both real-world train and val subset sizes must be set")

    samples_path = Path(config.realworld_dataset_dir) / "samples.npz"
    with np.load(samples_path, allow_pickle=False) as samples:
        gt_rmse = samples["gt_rmse"]

    rng = np.random.default_rng(config.seed)
    val_indices, _ = create_stratified_subset(
        gt_rmse,
        config.realworld_val_subset_size,
        _DEFAULT_BIN_EDGES,
        rng,
    )
    remaining_mask = np.ones(len(gt_rmse), dtype=bool)
    remaining_mask[val_indices] = False
    remaining_indices = np.where(remaining_mask)[0]

    if config.realworld_train_subset_size > len(remaining_indices):
        raise ValueError(
            "Requested real-world training subset is larger than the remaining "
            "sample pool after validation split"
        )

    train_rel_indices, _ = create_stratified_subset(
        gt_rmse[remaining_indices],
        config.realworld_train_subset_size,
        _DEFAULT_BIN_EDGES,
        rng,
    )
    train_indices = remaining_indices[train_rel_indices]
    return train_indices.astype(np.int64), val_indices.astype(np.int64)


def train(config: TrainConfig):
    """Run training loop."""
    # Seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {config.arch} on {device}")

    # Model
    if config.arch == "option_e":
        model = BoundaryTransformer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            n_points=config.n_points,
        )
    elif config.arch == "point_baseline":
        model = PointSetBaseline(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )
    else:
        model = SDFCropCNN(crop_size=config.crop_size)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Datasets
    train_ds, val_ds, train_source, val_source = build_datasets(config)
    print(f"Training data source: {train_source}")
    print(f"Validation data source: {val_source}")
    train_num_workers = resolve_num_workers(config.num_workers)
    val_num_workers = resolve_num_workers(min(2, config.num_workers))
    pin_memory = device.type == "cuda"
    collate_fn = collate_skip_none if train_source == "real-world" else None
    val_collate_fn = collate_skip_none if val_source == "real-world" else None

    train_loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": train_num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if not isinstance(train_ds, SyntheticHomographyDataset):
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        collate_fn=val_collate_fn,
    )

    # Output directory
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    best_metrics = {}
    patience_counter = 0
    training_log = []
    log_interval = max(1, config.steps_per_epoch // 20)  # ~20 updates per epoch

    t_train_start = time.time()

    for epoch in range(config.epochs):
        t_epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        window_loss_sum = 0.0  # loss over the last log_interval steps
        window_steps = 0

        train_iter = iter(train_loader)
        for step in range(config.steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if batch is None:
                continue

            pred_rmse, log_sigma, gt_rmse, censored = forward_model_batch(
                model,
                config.arch,
                batch,
                device,
            )

            loss = censored_heteroscedastic_loss(
                pred_rmse,
                log_sigma,
                gt_rmse,
                config.n_clip,
                censored=censored,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            loss_val = loss.item()
            train_loss_sum += loss_val
            train_steps += 1
            window_loss_sum += loss_val
            window_steps += 1

            if (step + 1) % log_interval == 0:
                now = time.time()
                window_loss = window_loss_sum / window_steps
                elapsed_epoch = now - t_epoch_start
                steps_done = step + 1
                sps = steps_done / elapsed_epoch  # steps/sec this epoch
                eta_epoch = (config.steps_per_epoch - steps_done) / max(sps, 1e-6)
                epochs_done = epoch + steps_done / config.steps_per_epoch
                elapsed_total = now - t_train_start
                eta_total = (
                    elapsed_total
                    / max(epochs_done, 1e-6)
                    * (config.epochs - epochs_done)
                )
                print(
                    f"  ep {epoch + 1}/{config.epochs}"
                    f"  step {steps_done}/{config.steps_per_epoch}"
                    f"  loss={window_loss:.4f}"
                    f"  {sps:.1f} steps/s"
                    f"  eta_epoch={eta_epoch / 60:.1f}m"
                    f"  eta_total={eta_total / 3600:.2f}h"
                )
                window_loss_sum = 0.0
                window_steps = 0

        train_loss_avg = train_loss_sum / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        all_pred_rmse = []
        all_gt_rmse = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                pred_rmse, log_sigma, gt_rmse, censored = forward_model_batch(
                    model,
                    config.arch,
                    batch,
                    device,
                )

                loss = censored_heteroscedastic_loss(
                    pred_rmse,
                    log_sigma,
                    gt_rmse,
                    config.n_clip,
                    censored=censored,
                )

                val_loss_sum += loss.item() * len(gt_rmse)
                val_steps += len(gt_rmse)

                all_pred_rmse.append(pred_rmse.cpu().numpy())
                all_gt_rmse.append(gt_rmse.cpu().numpy())

        val_loss_avg = val_loss_sum / max(val_steps, 1)
        if not all_pred_rmse:
            raise ValueError("Validation produced no valid samples")
        all_pred = np.concatenate(all_pred_rmse)
        all_gt = np.concatenate(all_gt_rmse)
        metrics = compute_val_metrics(all_pred, all_gt)

        elapsed = time.time() - t_epoch_start
        sps_epoch = config.steps_per_epoch / max(elapsed, 1e-6)
        print(
            f"Epoch {epoch + 1}/{config.epochs}  "
            f"train={train_loss_avg:.4f}  val={val_loss_avg:.4f}  "
            f"acc@5={metrics['accuracy_5px']:.3f}  "
            f"acc@15={metrics['accuracy_15px']:.3f}  "
            f"bal@5={metrics['balanced_acc_5px']:.3f}  "
            f"bal@15={metrics['balanced_acc_15px']:.3f}  "
            f"rho@15={metrics['spearman_rho_15px']:.3f}  "
            f"{sps_epoch:.1f} steps/s  {elapsed:.0f}s"
        )

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            **metrics,
        }
        training_log.append(epoch_log)

        # Checkpoint
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_metrics = metrics
            patience_counter = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "config": asdict(config),
                    "metrics": best_metrics,
                },
                out_dir / "best.pt",
            )
            print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(
                    f"  Early stopping after {config.patience} epochs "
                    f"without improvement."
                )
                break

        scheduler.step()

    # Save final model + log
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "config": asdict(config),
            "metrics": best_metrics,
        },
        out_dir / "final.pt",
    )

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")
    print(f"Best metrics: {best_metrics}")


def main():
    parser = argparse.ArgumentParser(
        description="Train NN homography quality estimator"
    )
    parser.add_argument(
        "--arch", default="option_e", choices=["option_e", "option_c", "point_baseline"]
    )
    parser.add_argument("--output-dir", default="checkpoints/nn_quality")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=10_000)
    parser.add_argument("--val-steps", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--sdf-downsample",
        type=int,
        default=4,
        help="Downsample factor for SDF computation (1=full, 4=16x faster)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Pre-generated dataset directory (from generate_dataset.py). "
        "If set, skips on-the-fly generation entirely.",
    )
    parser.add_argument(
        "--realworld-dataset-dir",
        default=None,
        help="Collected real-world dataset directory. If set, validation uses this dataset.",
    )
    parser.add_argument(
        "--realworld-train-subset-size",
        type=int,
        default=None,
        help="Optional real-world subset size to use for training.",
    )
    parser.add_argument(
        "--realworld-val-subset-size",
        type=int,
        default=None,
        help="Optional real-world subset size to use for validation.",
    )
    parser.add_argument(
        "--realworld-train-augment",
        action="store_true",
        help="Apply synthetic mask corruption augmentations to real-world training masks.",
    )

    args = parser.parse_args()
    config = TrainConfig(
        arch=args.arch,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_points=args.n_points,
        crop_size=args.crop_size,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
        sdf_downsample=args.sdf_downsample,
        cache_dir=args.cache_dir,
        realworld_dataset_dir=args.realworld_dataset_dir,
        realworld_train_subset_size=args.realworld_train_subset_size,
        realworld_val_subset_size=args.realworld_val_subset_size,
        realworld_train_augment=args.realworld_train_augment,
    )
    train(config)


if __name__ == "__main__":
    main()
