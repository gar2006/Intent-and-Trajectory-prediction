from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_phase4 import evaluate
from trajectory_baseline.phase4_utils import DEFAULT_DATA_ROOT, INTENT_NAMES


def save_label_distribution(label_counts: list[int], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(INTENT_NAMES, label_counts, color=["#1b9e77", "#d95f02", "#7570b3", "#66a61e"])
    ax.set_title("Intent Label Distribution")
    ax.set_ylabel("Sample Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(confusion_matrix: list[list[int]], output_path: Path) -> None:
    matrix = np.asarray(confusion_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(INTENT_NAMES)), INTENT_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(INTENT_NAMES)), INTENT_NAMES)
    ax.set_title("Intent Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_metric_summary(metrics: dict, output_path: Path) -> None:
    names = ["Oracle minADE", "Oracle minFDE", "Top-1 ADE", "Top-1 FDE"]
    values = [
        metrics["oracle_minADE"],
        metrics["oracle_minFDE"],
        metrics["top1_ADE"],
        metrics["top1_FDE"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color=["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"])
    ax.set_title("Trajectory Error Summary")
    ax.set_ylabel("Meters")
    ax.grid(axis="y", alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 4 evaluation report assets")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phase4_multimodal_best.pt",
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--output-dir", type=str, default="outputs/phase4_report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    metrics_path = output_dir / f"metrics_{args.split}.json"
    label_plot_path = output_dir / f"intent_distribution_{args.split}.png"
    confusion_path = output_dir / f"confusion_matrix_{args.split}.png"
    errors_path = output_dir / f"trajectory_metrics_{args.split}.png"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_label_distribution(metrics["label_counts"], label_plot_path)
    save_confusion_matrix(metrics["confusion_matrix"], confusion_path)
    save_metric_summary(metrics, errors_path)

    print(f"Wrote report assets to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
