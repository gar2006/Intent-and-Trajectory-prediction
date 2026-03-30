from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from trajectory_baseline.phase4_utils import (
    DEFAULT_DATA_ROOT,
    INTENT_NAMES,
    build_phase4_dataset,
    build_phase4_loader,
    load_phase4_checkpoint,
)


def evaluate(
    checkpoint_path: str,
    data_root: str,
    split: str,
    batch_size: int | None,
    cache_dir: str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_phase4_checkpoint(checkpoint_path, device)
    dataset = build_phase4_dataset(
        data_root=data_root,
        split=split,
        checkpoint_args=checkpoint["args"],
        cache_dir=cache_dir,
    )
    loader = build_phase4_loader(dataset, checkpoint["args"], batch_size=batch_size)

    total = 0
    sum_oracle_ade = 0.0
    sum_oracle_fde = 0.0
    sum_top1_ade = 0.0
    sum_top1_fde = 0.0
    sum_oracle_miss = 0.0
    sum_top1_miss = 0.0
    sum_intent_correct = 0.0
    confusion = np.zeros((len(INTENT_NAMES), len(INTENT_NAMES)), dtype=np.int64)
    label_counts = np.zeros((len(INTENT_NAMES),), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split}", leave=False):
            history = batch["history"].to(device)
            map_patch = batch["map_patch"].to(device)
            neighbors = batch["neighbors"].to(device)
            neighbor_mask = batch["neighbor_mask"].to(device)
            agent_type_id = batch["agent_type_id"].to(device)
            future = batch["future"].to(device)
            intent = batch["intent"].to(device)

            outputs = model(history, map_patch, neighbors, neighbor_mask, agent_type_id)
            trajectories = outputs["trajectories"]
            mode_probs = torch.softmax(outputs["mode_logits"], dim=-1)
            target = future.unsqueeze(1)

            ade_per_mode = torch.norm(trajectories - target, dim=-1).mean(dim=-1)
            fde_per_mode = torch.norm(trajectories[:, :, -1] - future[:, None, -1], dim=-1)

            oracle_ade = ade_per_mode.min(dim=1).values
            oracle_fde = fde_per_mode.min(dim=1).values

            top1_idx = mode_probs.argmax(dim=1)
            batch_idx = torch.arange(history.size(0), device=device)
            top1_ade = ade_per_mode[batch_idx, top1_idx]
            top1_fde = fde_per_mode[batch_idx, top1_idx]

            pred_intent = outputs["intent_logits"].argmax(dim=1)

            batch_size_actual = history.size(0)
            total += batch_size_actual
            sum_oracle_ade += oracle_ade.sum().item()
            sum_oracle_fde += oracle_fde.sum().item()
            sum_top1_ade += top1_ade.sum().item()
            sum_top1_fde += top1_fde.sum().item()
            sum_oracle_miss += (oracle_fde > 2.0).float().sum().item()
            sum_top1_miss += (top1_fde > 2.0).float().sum().item()
            sum_intent_correct += (pred_intent == intent).float().sum().item()

            for true_idx, pred_idx in zip(intent.cpu().tolist(), pred_intent.cpu().tolist()):
                confusion[true_idx, pred_idx] += 1
                label_counts[true_idx] += 1

    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "split": split,
        "samples": total,
        "oracle_minADE": sum_oracle_ade / total,
        "oracle_minFDE": sum_oracle_fde / total,
        "oracle_miss_rate": sum_oracle_miss / total,
        "top1_ADE": sum_top1_ade / total,
        "top1_FDE": sum_top1_fde / total,
        "top1_miss_rate": sum_top1_miss / total,
        "intent_accuracy": sum_intent_correct / total,
        "intent_labels": INTENT_NAMES,
        "label_counts": label_counts.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phase4_multimodal_best.pt",
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    print(f"Checkpoint: {metrics['checkpoint']}")
    print(f"Split: {metrics['split']}")
    print(f"Samples: {metrics['samples']}")
    print(f"Oracle minADE: {metrics['oracle_minADE']:.4f}")
    print(f"Oracle minFDE: {metrics['oracle_minFDE']:.4f}")
    print(f"Oracle Miss Rate: {metrics['oracle_miss_rate']:.4f}")
    print(f"Top-1 ADE: {metrics['top1_ADE']:.4f}")
    print(f"Top-1 FDE: {metrics['top1_FDE']:.4f}")
    print(f"Top-1 Miss Rate: {metrics['top1_miss_rate']:.4f}")
    print(f"Intent Accuracy: {metrics['intent_accuracy']:.4f}")
    print("Intent labels:", metrics["intent_labels"])
    print("Intent label counts:", metrics["label_counts"])
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.asarray(metrics["confusion_matrix"]))

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"Wrote metrics JSON to {json_path.resolve()}")


if __name__ == "__main__":
    main()
