from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajectory_baseline.phase4_utils import DEFAULT_DATA_ROOT, INTENT_NAMES
from trajectory_baseline.dataset import NuScenesPedestrianDataset


DEFAULT_PREDICTIONS_PATH = Path("outputs/phase4_predictions_val.json")
DEFAULT_OUTPUT_PATH = Path("demo_site/data/demo_data.json")


INTENT_COPY = {
    "crossing": "may move sideways or cross across the scene",
    "waiting": "may slow down or pause",
    "turning": "may change direction",
    "walking_straight": "is likely to keep moving forward",
}

RISK_COPY = {
    "low": "Low risk",
    "medium": "Medium risk",
    "high": "High risk",
}


def describe_mode(rank: int, probability: float) -> str:
    labels = ["Most likely path", "Second possible path", "Third possible path"]
    return f"{labels[rank]} ({probability * 100:.0f}% confidence)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build website-friendly demo data from predictions")
    parser.add_argument("--predictions", type=str, default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--heading-aligned", action="store_true")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_path = Path(args.output)

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    pred_index = {
        (row["sample_token"], row["agent_id"]): row
        for row in predictions
    }

    dataset = NuScenesPedestrianDataset(
        data_root=args.data_root,
        split=args.split,
        cache_dir=args.cache_dir,
        include_map=False,
        include_social=False,
        include_intent=True,
        heading_aligned=args.heading_aligned,
    )

    records: list[dict] = []
    for item in dataset:
        key = (item["sample_token"], item["agent_id"])
        pred = pred_index.get(key)
        if pred is None:
            continue

        mode_probs = pred["mode_probabilities"]
        sorted_mode_indices = sorted(
            range(len(mode_probs)),
            key=lambda idx: mode_probs[idx],
            reverse=True,
        )

        ranked_modes = []
        for rank, mode_idx in enumerate(sorted_mode_indices):
            ranked_modes.append(
                {
                    "mode_index": mode_idx,
                    "label": describe_mode(rank, mode_probs[mode_idx]),
                    "probability": mode_probs[mode_idx],
                    "trajectory": pred["trajectories"][mode_idx],
                    "endpoint": pred["endpoints"][mode_idx],
                }
            )

        predicted_intent = pred["predicted_intent"]
        top_prob = mode_probs[sorted_mode_indices[0]]
        agent_type = item["agent_type"]
        readable_agent_type = agent_type.replace("_", " ").title()
        summary = (
            f"This {agent_type} {INTENT_COPY[predicted_intent]}. "
            f"The model's most likely future has about {top_prob * 100:.0f}% confidence, "
            f"with an overall {pred['risk_level']} risk reading."
        )

        records.append(
            {
                "sample_token": item["sample_token"],
                "agent_id": item["agent_id"],
                "agent_type": agent_type,
                "agent_type_label": readable_agent_type,
                "predicted_intent": predicted_intent,
                "predicted_intent_label": predicted_intent.replace("_", " ").title(),
                "risk_level": pred["risk_level"],
                "risk_label": RISK_COPY[pred["risk_level"]],
                "risk_score": pred["risk_score"],
                "risk_reasons": pred["risk_reasons"],
                "risk_factors": pred["risk_factors"],
                "intent_probabilities": [
                    {"intent": name, "probability": prob}
                    for name, prob in zip(INTENT_NAMES, pred["intent_probabilities"])
                ],
                "history_xy": item["history_xy"].tolist(),
                "future_xy": item["future"].tolist(),
                "ranked_modes": ranked_modes,
                "summary": summary,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} demo records to {output_path.resolve()}")


if __name__ == "__main__":
    main()
