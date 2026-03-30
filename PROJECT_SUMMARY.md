# Vulnerable Road User Trajectory Prediction for nuScenes Mini

## Goal

Build a hackathon-ready prediction system that forecasts future motion for:

- pedestrians
- cyclists

The final system predicts:

- `K=3` future trajectories over 6 seconds
- a confidence score for each trajectory
- an intent label

## Final Pipeline

### Inputs per agent

- Past `(x, y)` positions over 4 timesteps at 2 Hz
- Derived kinematics: velocity and acceleration
- Local `100x100` map patch
- Nearby vulnerable road users within `10m`

### Model

- Transformer encoder for trajectory history
- CNN encoder for local map patch
- GAT-style social encoder for nearby road users
- Shared fused scene embedding
- Intent classification head
- `K=3` endpoint proposals
- Goal-conditioned multimodal trajectory decoder
- Probability head over the 3 modes

## What Was Built

- Raw nuScenes JSON data loader
- Pedestrian + cyclist extraction
- Map patch pipeline
- Social interaction modeling
- Multimodal `K=3` trajectory forecasting
- Intent classification
- Evaluation, export, visualization, and website demo

## Final Training Setup

We tested multiple checkpoints and selected the **agent-type-aware 20-epoch** model as the final one.

Why this model was chosen:

- it adds an explicit learned embedding for `pedestrian` vs `cyclist`
- it dramatically improves the quality of the **top-ranked** future
- it also improves intent accuracy
- oracle metrics remain strong enough for a reliable multimodal forecast

## Final Chosen Checkpoint

- [phase4_multimodal_best.pt](/Users/chaku/Desktop/Mobility%20challenge/checkpoints/vru_manualsplit_intent_type/phase4_multimodal_best.pt)

## Final Validation Metrics

Metrics on the corrected manual validation split:

| Metric | Value |
|---|---:|
| Oracle minADE | 0.8392 |
| Oracle minFDE | 1.6619 |
| Oracle Miss Rate | 0.1809 |
| Top-1 ADE | 1.0384 |
| Top-1 FDE | 2.0585 |
| Top-1 Miss Rate | 0.2714 |
| Intent Accuracy | 0.7720 |

## Validation Split

We replaced the original random mini split with a manual scene split because:

- `scene-1077` contains no pedestrian data
- random mini splits were producing an unrealistic validation set
- the corrected split gives meaningful class diversity

Final validation label counts:

- `crossing = 68`
- `waiting = 371`
- `turning = 101`
- `walking_straight = 267`

## Best Final Assets

- Metrics JSON: [metrics_val.json](/Users/chaku/Desktop/Mobility%20challenge/outputs/phase4_report_type_final/metrics_val.json)
- Trajectory metrics chart: [trajectory_metrics_val.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/phase4_report_type_final/trajectory_metrics_val.png)
- Intent distribution: [intent_distribution_val.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/phase4_report_type_final/intent_distribution_val.png)
- Confusion matrix: [confusion_matrix_val.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/phase4_report_type_final/confusion_matrix_val.png)
- Demo predictions JSON: [final_predictions_type.json](/Users/chaku/Desktop/Mobility%20challenge/outputs/final_predictions_type.json)
- Demo visualizations: [val_sample_0000.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/visualizations_type_final/val_sample_0000.png)

## Best Demo Story

- The system predicts multiple possible futures instead of one fixed future
- It uses motion history, map context, and nearby road users
- It works for both pedestrians and cyclists and now models them differently
- Oracle metrics show the model still generates strong alternative futures
- Top-1 metrics improved sharply once we added the agent-type embedding

## Honest Caveat

Intent classification is much better now, but it is still not perfect:

- `turning` remains harder than `waiting`
- the model still confuses some crossing and turning cases
- map semantics and richer social features are the best next improvements

This is still a strong prototype and a meaningful end-to-end result for the hackathon.
