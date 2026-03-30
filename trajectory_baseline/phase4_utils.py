from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trajectory_baseline.dataset import NuScenesPedestrianDataset
from trajectory_baseline.model import TransformerMapSocialMultiModalPredictor


DEFAULT_DATA_ROOT = "/Users/chaku/Desktop/Mobility challenge/v1.0-mini"
INTENT_NAMES = ["crossing", "waiting", "turning", "walking_straight"]


def build_phase4_dataset(
    data_root: str,
    split: str,
    checkpoint_args: dict,
    cache_dir: str,
) -> NuScenesPedestrianDataset:
    return NuScenesPedestrianDataset(
        data_root=data_root,
        split=split,
        past_steps=checkpoint_args.get("past_steps", 4),
        future_steps=checkpoint_args.get("future_steps", 12),
        split_seed=checkpoint_args.get("seed", 42),
        cache_dir=cache_dir,
        include_map=True,
        map_patch_size=checkpoint_args.get("map_patch_size", 100),
        include_social=True,
        social_radius=checkpoint_args.get("social_radius", 10.0),
        max_neighbors=checkpoint_args.get("max_neighbors", 8),
        include_intent=True,
        heading_aligned=checkpoint_args.get("heading_aligned", False),
        include_cyclists=checkpoint_args.get("include_cyclists", True),
        use_manual_mini_split=checkpoint_args.get("use_manual_mini_split", True),
    )


def build_phase4_model(checkpoint_args: dict, device: torch.device) -> TransformerMapSocialMultiModalPredictor:
    model = TransformerMapSocialMultiModalPredictor(
        input_dim=6,
        model_dim=checkpoint_args.get("model_dim", 128),
        num_heads=checkpoint_args.get("num_heads", 4),
        num_layers=checkpoint_args.get("num_layers", 2),
        ff_dim=checkpoint_args.get("ff_dim", 256),
        past_steps=checkpoint_args.get("past_steps", 4),
        future_steps=checkpoint_args.get("future_steps", 12),
        num_modes=checkpoint_args.get("num_modes", 3),
        num_intents=len(INTENT_NAMES),
        dropout=checkpoint_args.get("dropout", 0.1),
        social_encoder_type=checkpoint_args.get("social_encoder_type", "gat"),
        social_pooling_type=checkpoint_args.get("social_pooling_type", "mean"),
        use_agent_type_embedding=checkpoint_args.get("use_agent_type_embedding", False),
    ).to(device)
    return model


def load_phase4_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TransformerMapSocialMultiModalPredictor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_phase4_model(checkpoint["args"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_phase4_loader(
    dataset: NuScenesPedestrianDataset,
    checkpoint_args: dict,
    batch_size: int | None = None,
    shuffle: bool = False,
) -> DataLoader:
    resolved_batch_size = batch_size or checkpoint_args.get("batch_size", 64)
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        num_workers=checkpoint_args.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
