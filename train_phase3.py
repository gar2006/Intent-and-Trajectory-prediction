from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajectory_baseline.dataset import NuScenesPedestrianDataset
from trajectory_baseline.model import TransformerMapSocialTrajectoryPredictor


DEFAULT_DATA_ROOT = "/Users/chaku/Downloads/v1.0-mini"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target, dim=-1).mean(dim=-1)


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred[:, -1] - target[:, -1], dim=-1)


def miss_rate(pred: torch.Tensor, target: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    return (fde(pred, target) > threshold).float()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_miss = 0.0
    total_examples = 0

    for batch in tqdm(loader, leave=False):
        history = batch["history"].to(device)
        map_patch = batch["map_patch"].to(device)
        neighbors = batch["neighbors"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)
        future = batch["future"].to(device)

        pred = model(history, map_patch, neighbors, neighbor_mask)
        loss = nn.functional.mse_loss(pred, future)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = history.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_ade += ade(pred, future).sum().item()
        total_fde += fde(pred, future).sum().item()
        total_miss += miss_rate(pred, future).sum().item()

    return {
        "loss": total_loss / total_examples,
        "ade": total_ade / total_examples,
        "fde": total_fde / total_examples,
        "miss_rate": total_miss / total_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Transformer + map + social baseline")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--past-steps", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--map-patch-size", type=int, default=100)
    parser.add_argument("--social-radius", type=float, default=10.0)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NuScenesPedestrianDataset(
        data_root=args.data_root,
        split="train",
        past_steps=args.past_steps,
        future_steps=args.future_steps,
        split_seed=args.seed,
        cache_dir=args.cache_dir,
        include_map=True,
        map_patch_size=args.map_patch_size,
        include_social=True,
        social_radius=args.social_radius,
        max_neighbors=args.max_neighbors,
    )
    val_dataset = NuScenesPedestrianDataset(
        data_root=args.data_root,
        split="val",
        past_steps=args.past_steps,
        future_steps=args.future_steps,
        split_seed=args.seed,
        cache_dir=args.cache_dir,
        include_map=True,
        map_patch_size=args.map_patch_size,
        include_social=True,
        social_radius=args.social_radius,
        max_neighbors=args.max_neighbors,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TransformerMapSocialTrajectoryPredictor(
        input_dim=6,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_ade = float("inf")

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_ADE={train_metrics['ade']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ADE={val_metrics['ade']:.4f} "
            f"val_FDE={val_metrics['fde']:.4f} "
            f"val_MR={val_metrics['miss_rate']:.4f}"
        )

        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            checkpoint_path = save_dir / "phase3_transformer_map_social_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
