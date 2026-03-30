# Demo Script

## Demo Goal

Show that the system:

- forecasts future motion for pedestrians and cyclists
- predicts multiple futures rather than a single rigid answer
- uses map context and nearby road users
- can be explained visually to a non-technical audience

## Final Model Used in the Demo

- [phase4_multimodal_best.pt](/Users/chaku/Desktop/Mobility%20challenge/checkpoints/vru_manualsplit_intent_type/phase4_multimodal_best.pt)

## 3-Minute Demo Flow

### 1. Open the project summary

Show:

- [PROJECT_SUMMARY.md](/Users/chaku/Desktop/Mobility%20challenge/PROJECT_SUMMARY.md)

Say:

`We built an end-to-end vulnerable road user forecasting system on nuScenes mini that works for both pedestrians and cyclists.`

### 2. Show the headline metrics

Show:

- [SLIDE_METRICS.md](/Users/chaku/Desktop/Mobility%20challenge/SLIDE_METRICS.md)
- [trajectory_metrics_val.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/phase4_report_type_final/trajectory_metrics_val.png)

Say:

`We selected the 20-epoch type-aware checkpoint because it gave the best overall practical balance of forecasting quality, confidence ranking, and intent classification.`

`Oracle metrics are still strong, and once we added an explicit pedestrian-versus-cyclist embedding, the top-ranked prediction improved a lot as well.`

### 3. Show one final visualization

Show:

- [val_sample_0000.png](/Users/chaku/Desktop/Mobility%20challenge/outputs/visualizations_type_final/val_sample_0000.png)

Say:

`Black is the observed history, green is the actual future, and the colored lines are the three predicted futures.`

`This helps a non-technical viewer understand both the uncertainty and the model confidence.`

### 4. Show the website demo

Run:

```bash
cd "/Users/chaku/Desktop/Mobility challenge"
python3 -m http.server 8000
```

Open:

- [http://127.0.0.1:8000/demo_site/](http://127.0.0.1:8000/demo_site/)

Say:

`The website presents the output in simple language, clearly distinguishes between pedestrians and cyclists, and now includes a simple risk reading.`

### 5. Close with the final positioning

Say:

`This is a working multimodal trajectory prediction prototype with map and social awareness, and we validated multiple training regimes before choosing the final type-aware model.`

`The biggest next step would be improving turning behavior, richer map semantics, and stronger social interaction features.`

## Useful Final Commands

### Evaluate the chosen checkpoint

```bash
cd "/Users/chaku/Desktop/Mobility challenge"
python3 evaluate_phase4.py \
  --checkpoint checkpoints/vru_manualsplit_intent_type/phase4_multimodal_best.pt \
  --split val \
  --batch-size 32
```

### Export the chosen checkpoint predictions

```bash
cd "/Users/chaku/Desktop/Mobility challenge"
python3 predict_phase4.py \
  --checkpoint checkpoints/vru_manualsplit_intent_type/phase4_multimodal_best.pt \
  --split val \
  --output outputs/final_predictions_type.json
```

### Rebuild the website data

```bash
cd "/Users/chaku/Desktop/Mobility challenge"
python3 build_demo_data.py \
  --predictions outputs/final_predictions_type.json \
  --output demo_site/data/demo_data.json \
  --split val \
  --heading-aligned
```
