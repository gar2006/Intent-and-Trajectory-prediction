from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from trajectory_baseline.phase4_utils import (
    DEFAULT_DATA_ROOT,
    INTENT_NAMES,
    build_phase4_dataset,
    load_phase4_checkpoint,
)


MODE_COLORS = ["#d73027", "#fc8d59", "#4575b4"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Phase 4 predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phase4_multimodal_best.pt",
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations_phase4",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_phase4_checkpoint(args.checkpoint, device)
    dataset = build_phase4_dataset(
        data_root=args.data_root,
        split=args.split,
        checkpoint_args=checkpoint["args"],
        cache_dir=args.cache_dir,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    end_index = min(len(dataset), args.start_index + args.num_samples)
    for idx in range(args.start_index, end_index):
        item = dataset[idx]
        history = item["history"].unsqueeze(0).to(device)
        map_patch = item["map_patch"].unsqueeze(0).to(device)
        neighbors = item["neighbors"].unsqueeze(0).to(device)
        neighbor_mask = item["neighbor_mask"].unsqueeze(0).to(device)
        agent_type_id = item["agent_type_id"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(history, map_patch, neighbors, neighbor_mask, agent_type_id)
            mode_probs = torch.softmax(outputs["mode_logits"], dim=-1)[0].cpu()
            intent_probs = torch.softmax(outputs["intent_logits"], dim=-1)[0].cpu()
            trajectories = outputs["trajectories"][0].cpu()

        map_image = item["map_patch"][0].cpu().numpy()
        history_xy = item["history_xy"].cpu().numpy()
        future_xy = item["future"].cpu().numpy()
        predicted_intent = INTENT_NAMES[int(intent_probs.argmax().item())]
        true_intent = INTENT_NAMES[int(item["intent"].item())]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            map_image,
            cmap="gray",
            origin="lower",
            extent=[-15, 15, -15, 15],
            alpha=0.75,
        )

        ax.plot(history_xy[:, 0], history_xy[:, 1], "ko-", linewidth=2, label="Observed")
        ax.plot(future_xy[:, 0], future_xy[:, 1], color="#1a9850", linewidth=2.5, label="Ground truth")

        for mode_idx in range(trajectories.size(0)):
            traj = trajectories[mode_idx].numpy()
            prob = mode_probs[mode_idx].item()
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=MODE_COLORS[mode_idx % len(MODE_COLORS)],
                linewidth=2,
                alpha=0.9,
                label=f"Mode {mode_idx + 1} ({prob:.2f})",
            )
            ax.scatter(traj[-1, 0], traj[-1, 1], color=MODE_COLORS[mode_idx % len(MODE_COLORS)], s=30)

        ax.scatter(0.0, 0.0, color="black", s=45, marker="x")
        ax.set_title(
            f"Sample {idx} | true={true_intent} | pred={predicted_intent}",
            fontsize=10,
        )
        ax.set_xlabel("Local x (m)")
        ax.set_ylabel("Local y (m)")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        out_path = output_dir / f"{args.split}_sample_{idx:04d}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    print(f"Saved {end_index - args.start_index} visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
