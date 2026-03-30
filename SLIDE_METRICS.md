# Slide Metrics

## One-Slide Summary

**Project:** Vulnerable Road User Trajectory Prediction on nuScenes Mini

**Agents:** Pedestrians + Cyclists

**Model:** Transformer history encoder + CNN map encoder + GAT-style social encoder + multimodal decoder + agent-type embedding

**Outputs:** `K=3` future trajectories, trajectory probabilities, intent class

| Category | Metric | Value |
|---|---|---:|
| Forecasting | Oracle minADE | 0.8392 |
| Forecasting | Oracle minFDE | 1.6619 |
| Forecasting | Oracle Miss Rate | 0.1809 |
| Forecasting | Top-1 ADE | 1.0384 |
| Forecasting | Top-1 FDE | 2.0585 |
| Forecasting | Top-1 Miss Rate | 0.2714 |
| Intent | Intent Accuracy | 0.7720 |

## Final Model Choice

**Chosen checkpoint:** 20 epochs + agent-type embedding

Why this replaced the old final checkpoint:

- oracle metrics stayed competitive
- top-1 forecasting improved dramatically
- intent accuracy improved as well
- explicit pedestrian/cyclist conditioning made the model more practical

## Speaker Notes

- We predict three plausible futures instead of only one.
- Oracle metrics tell us whether at least one predicted future is good.
- Top-1 metrics tell us whether the highest-confidence future is good.
- This final version sharply reduced the oracle vs top-1 gap.
- The system now supports both pedestrians and cyclists with an explicit type-aware embedding.

## Visuals To Show

- [trajectory_metrics_val.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/phase4_report_type_final/trajectory_metrics_val.png)
- [confusion_matrix_val.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/phase4_report_type_final/confusion_matrix_val.png)
- [val_sample_0000.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/visualizations_type_final/val_sample_0000.png)
