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
from trajectory_baseline.model import TransformerMapSocialMultiModalPredictor


DEFAULT_DATA_ROOT = "/Users/chaku/Desktop/Mobility challenge/v1.0-mini"
INTENT_NAMES = ["crossing", "waiting", "turning", "walking_straight"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trajectory_errors(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_expanded = target.unsqueeze(1)
    ade = torch.norm(pred - target_expanded, dim=-1).mean(dim=-1)
    fde = torch.norm(pred[:, :, -1] - target[:, None, -1], dim=-1)
    return ade, fde


def compute_losses(
    outputs: dict[str, torch.Tensor],
    future: torch.Tensor,
    intent: torch.Tensor,
    intent_weight: float,
    endpoint_weight: float,
    nll_weight: float,
    diversity_weight: float,
    probability_temperature: float,
    intent_label_smoothing: float,
    intent_class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    trajectories = outputs["trajectories"]
    endpoints = outputs["endpoints"]
    mode_logits = outputs["mode_logits"]
    intent_logits = outputs["intent_logits"]

    ade_per_mode, fde_per_mode = trajectory_errors(trajectories, future)
    best_mode = ade_per_mode.argmin(dim=1)
    batch_indices = torch.arange(future.size(0), device=future.device)

    minade_loss = ade_per_mode[batch_indices, best_mode].mean()
    endpoint_target = future[:, -1]
    endpoint_loss = torch.norm(
        endpoints[batch_indices, best_mode] - endpoint_target,
        dim=-1,
    ).mean()
    intent_loss = nn.functional.cross_entropy(
        intent_logits,
        intent,
        weight=intent_class_weights,
        label_smoothing=intent_label_smoothing,
    )

    soft_targets = torch.softmax(-ade_per_mode / max(probability_temperature, 1e-6), dim=1)
    nll_loss = nn.functional.kl_div(
        nn.functional.log_softmax(mode_logits, dim=1),
        soft_targets,
        reduction="batchmean",
    )

    pairwise_dist = torch.cdist(endpoints, endpoints, p=2)
    mode_count = endpoints.size(1)
    pair_mask = torch.triu(
        torch.ones((mode_count, mode_count), device=endpoints.device, dtype=torch.bool),
        diagonal=1,
    )
    diversity_penalty = torch.exp(-pairwise_dist[:, pair_mask]).mean()

    total_loss = minade_loss
    total_loss = total_loss + endpoint_weight * endpoint_loss
    total_loss = total_loss + intent_weight * intent_loss
    total_loss = total_loss + nll_weight * nll_loss
    total_loss = total_loss + diversity_weight * diversity_penalty

    return total_loss, {
        "minade_loss": minade_loss.detach(),
        "endpoint_loss": endpoint_loss.detach(),
        "intent_loss": intent_loss.detach(),
        "nll_loss": nll_loss.detach(),
        "diversity_loss": diversity_penalty.detach(),
        "best_mode": best_mode.detach(),
        "ade_per_mode": ade_per_mode.detach(),
        "fde_per_mode": fde_per_mode.detach(),
        "intent_logits": intent_logits.detach(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    intent_weight: float,
    endpoint_weight: float,
    nll_weight: float,
    diversity_weight: float,
    probability_temperature: float,
    intent_label_smoothing: float,
    intent_class_weights: torch.Tensor | None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_examples = 0
    total_loss = 0.0
    total_minade = 0.0
    total_minfde = 0.0
    total_miss = 0.0
    total_intent_loss = 0.0
    total_endpoint_loss = 0.0
    total_nll_loss = 0.0
    total_diversity_loss = 0.0
    total_intent_correct = 0.0

    for batch in tqdm(loader, leave=False):
        history = batch["history"].to(device)
        map_patch = batch["map_patch"].to(device)
        neighbors = batch["neighbors"].to(device)
        neighbor_mask = batch["neighbor_mask"].to(device)
        agent_type_id = batch["agent_type_id"].to(device)
        future = batch["future"].to(device)
        intent = batch["intent"].to(device)

        outputs = model(history, map_patch, neighbors, neighbor_mask, agent_type_id)
        loss, extras = compute_losses(
            outputs=outputs,
            future=future,
            intent=intent,
            intent_weight=intent_weight,
            endpoint_weight=endpoint_weight,
            nll_weight=nll_weight,
            diversity_weight=diversity_weight,
            probability_temperature=probability_temperature,
            intent_label_smoothing=intent_label_smoothing,
            intent_class_weights=intent_class_weights,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = history.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_minade += extras["ade_per_mode"].min(dim=1).values.sum().item()
        total_minfde += extras["fde_per_mode"].min(dim=1).values.sum().item()
        total_miss += (extras["fde_per_mode"].min(dim=1).values > 2.0).float().sum().item()
        total_endpoint_loss += extras["endpoint_loss"].item() * batch_size
        total_intent_loss += extras["intent_loss"].item() * batch_size
        total_nll_loss += extras["nll_loss"].item() * batch_size
        total_diversity_loss += extras["diversity_loss"].item() * batch_size
        total_intent_correct += (
            extras["intent_logits"].argmax(dim=1) == intent
        ).float().sum().item()

    return {
        "loss": total_loss / total_examples,
        "minADE": total_minade / total_examples,
        "minFDE": total_minfde / total_examples,
        "miss_rate": total_miss / total_examples,
        "intent_loss": total_intent_loss / total_examples,
        "endpoint_loss": total_endpoint_loss / total_examples,
        "nll_loss": total_nll_loss / total_examples,
        "diversity_loss": total_diversity_loss / total_examples,
        "intent_acc": total_intent_correct / total_examples,
    }


def build_intent_class_weights(dataset: NuScenesPedestrianDataset) -> torch.Tensor:
    counts = torch.zeros(len(INTENT_NAMES), dtype=torch.float32)
    for sample in dataset.samples:
        counts[sample.intent_label] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 multimodal trajectory + intent baseline")
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
    parser.add_argument("--num-modes", type=int, default=3)
    parser.add_argument("--intent-weight", type=float, default=0.5)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--nll-weight", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--probability-temperature", type=float, default=0.75)
    parser.add_argument("--intent-label-smoothing", type=float, default=0.05)
    parser.add_argument("--heading-aligned", action="store_true")
    parser.add_argument("--no-cyclists", action="store_true")
    parser.add_argument("--no-manual-mini-split", action="store_true")
    parser.add_argument("--social-encoder-type", type=str, default="gat", choices=["gat", "pool"])
    parser.add_argument("--social-pooling-type", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--use-agent-type-embedding", action="store_true")
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
        include_intent=True,
        heading_aligned=args.heading_aligned,
        include_cyclists=not args.no_cyclists,
        use_manual_mini_split=not args.no_manual_mini_split,
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
        include_intent=True,
        heading_aligned=args.heading_aligned,
        include_cyclists=not args.no_cyclists,
        use_manual_mini_split=not args.no_manual_mini_split,
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

    model = TransformerMapSocialMultiModalPredictor(
        input_dim=6,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
        num_modes=args.num_modes,
        num_intents=len(INTENT_NAMES),
        dropout=args.dropout,
        social_encoder_type=args.social_encoder_type,
        social_pooling_type=args.social_pooling_type,
        use_agent_type_embedding=args.use_agent_type_embedding,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    intent_class_weights = build_intent_class_weights(train_dataset).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_minade = float("inf")

    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Intent labels: {INTENT_NAMES}")
    print(f"Heading aligned: {args.heading_aligned}")
    print(f"Include cyclists: {not args.no_cyclists}")
    print(f"Manual mini split: {not args.no_manual_mini_split}")
    print(f"Social encoder: {args.social_encoder_type} ({args.social_pooling_type})")
    print(f"Agent-type embedding: {args.use_agent_type_embedding}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            intent_weight=args.intent_weight,
            endpoint_weight=args.endpoint_weight,
            nll_weight=args.nll_weight,
            diversity_weight=args.diversity_weight,
            probability_temperature=args.probability_temperature,
            intent_label_smoothing=args.intent_label_smoothing,
            intent_class_weights=intent_class_weights,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                intent_weight=args.intent_weight,
                endpoint_weight=args.endpoint_weight,
                nll_weight=args.nll_weight,
                diversity_weight=args.diversity_weight,
                probability_temperature=args.probability_temperature,
                intent_label_smoothing=0.0,
                intent_class_weights=intent_class_weights,
            )
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_minADE={train_metrics['minADE']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_minADE={val_metrics['minADE']:.4f} "
            f"val_minFDE={val_metrics['minFDE']:.4f} "
            f"val_MR={val_metrics['miss_rate']:.4f} "
            f"val_intent_acc={val_metrics['intent_acc']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_metrics["minADE"] < best_val_minade:
            best_val_minade = val_metrics["minADE"]
            checkpoint_path = save_dir / "phase4_multimodal_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "intent_names": INTENT_NAMES,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
