from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from trajectory_baseline.phase4_utils import (
    DEFAULT_DATA_ROOT,
    INTENT_NAMES,
    build_phase4_dataset,
    build_phase4_loader,
    load_phase4_checkpoint,
)


def compute_risk_summary(
    agent_type: str,
    predicted_intent: str,
    mode_probabilities: list[float],
    endpoints: list[list[float]],
) -> dict:
    top_probability = max(mode_probabilities)
    entropy = 0.0
    for probability in mode_probabilities:
        probability = max(probability, 1e-8)
        entropy -= probability * math.log(probability)
    entropy /= math.log(max(len(mode_probabilities), 2))

    pairwise = []
    for left_idx in range(len(endpoints)):
        for right_idx in range(left_idx + 1, len(endpoints)):
            dx = endpoints[left_idx][0] - endpoints[right_idx][0]
            dy = endpoints[left_idx][1] - endpoints[right_idx][1]
            pairwise.append(math.sqrt(dx * dx + dy * dy))
    endpoint_spread = sum(pairwise) / max(len(pairwise), 1)
    spread_score = min(endpoint_spread / 6.0, 1.0)

    intent_score = {
        "crossing": 0.85,
        "turning": 0.62,
        "walking_straight": 0.34,
        "waiting": 0.22,
    }[predicted_intent]
    type_bonus = 0.08 if agent_type == "cyclist" else 0.0
    confidence_gap = 1.0 - top_probability

    risk_score = (
        0.45 * intent_score
        + 0.23 * entropy
        + 0.2 * spread_score
        + 0.12 * confidence_gap
        + type_bonus
    )
    risk_score = max(0.0, min(risk_score, 1.0))

    if risk_score >= 0.68:
        risk_level = "high"
    elif risk_score >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    reasons: list[str] = []
    if predicted_intent == "crossing":
        reasons.append("The model expects lateral crossing-like movement.")
    elif predicted_intent == "turning":
        reasons.append("The forecast includes a meaningful direction change.")
    elif predicted_intent == "waiting":
        reasons.append("The road user may pause or hesitate near the current position.")
    else:
        reasons.append("The most likely path stays relatively consistent with recent motion.")

    if entropy >= 0.55 or top_probability <= 0.45:
        reasons.append("The model is uncertain because the three futures have similar confidence.")
    if spread_score >= 0.45:
        reasons.append("The predicted endpoints are spread apart, so several distinct futures remain plausible.")
    if agent_type == "cyclist":
        reasons.append("Cyclists can change position faster, so we apply a slightly more cautious risk prior.")

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 4),
        "risk_reasons": reasons[:3],
        "risk_factors": {
            "top_probability": round(top_probability, 4),
            "normalized_entropy": round(entropy, 4),
            "endpoint_spread": round(endpoint_spread, 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 4 predictions to JSON")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phase4_multimodal_best.pt",
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/phase4_predictions_val.json",
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
    loader = build_phase4_loader(dataset, checkpoint["args"], batch_size=args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting {args.split}", leave=False):
            history = batch["history"].to(device)
            map_patch = batch["map_patch"].to(device)
            neighbors = batch["neighbors"].to(device)
            neighbor_mask = batch["neighbor_mask"].to(device)
            agent_type_id = batch["agent_type_id"].to(device)

            outputs = model(history, map_patch, neighbors, neighbor_mask, agent_type_id)
            mode_probs = torch.softmax(outputs["mode_logits"], dim=-1).cpu()
            trajectories = outputs["trajectories"].cpu()
            endpoints = outputs["endpoints"].cpu()
            intent_probs = torch.softmax(outputs["intent_logits"], dim=-1).cpu()
            predicted_intent = intent_probs.argmax(dim=-1)

            for idx in range(history.size(0)):
                risk_summary = compute_risk_summary(
                    agent_type=batch["agent_type"][idx],
                    predicted_intent=INTENT_NAMES[predicted_intent[idx].item()],
                    mode_probabilities=mode_probs[idx].tolist(),
                    endpoints=endpoints[idx].tolist(),
                )
                records.append(
                    {
                        "sample_token": batch["sample_token"][idx],
                        "agent_id": batch["agent_id"][idx],
                        "agent_type": batch["agent_type"][idx],
                        "predicted_intent": INTENT_NAMES[predicted_intent[idx].item()],
                        "intent_probabilities": intent_probs[idx].tolist(),
                        "mode_probabilities": mode_probs[idx].tolist(),
                        "endpoints": endpoints[idx].tolist(),
                        "trajectories": trajectories[idx].tolist(),
                        **risk_summary,
                    }
                )

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)

    print(f"Wrote {len(records)} predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
