# nuScenes pedestrian trajectory baselines

This workspace now contains Phase 1 through Phase 5 for your hackathon:

- Phase 1: raw trajectory loader + encoder-decoder LSTM
- Phase 2: Transformer history encoder + local map patch CNN
- Phase 3: Transformer + map CNN + social GAT-style encoder
- Phase 4: multimodal `K=3` prediction + probabilities + intent classification
- Phase 5: evaluation, JSON export, and visualization scripts
- Common history features: `(x, y, vx, vy, ax, ay)` over 4 timesteps
- Future target: 12 future `(x, y)` points
- Training with MSE loss
- Validation metrics: ADE, FDE, Miss Rate

## Expected dataset layout

The current code is set up for your downloaded mini dataset at:

```text
/Users/chaku/Downloads/v1.0-mini
```

Inside that folder, the JSON tables should live here:

```text
/Users/chaku/Downloads/v1.0-mini/v1.0-mini/*.json
```

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Train Phase 1

```bash
python3 train_phase1.py --data-root /Users/chaku/Downloads/v1.0-mini --epochs 10
```

## Train Phase 2

```bash
python3 train_phase2.py --data-root /Users/chaku/Downloads/v1.0-mini --epochs 10
```

## Train Phase 3

```bash
python3 train_phase3.py --data-root /Users/chaku/Downloads/v1.0-mini --epochs 10
```

## Train Phase 4

```bash
python3 train_phase4.py --data-root /Users/chaku/Downloads/v1.0-mini --epochs 10
```

## Evaluate Phase 4

```bash
python3 evaluate_phase4.py --checkpoint checkpoints/phase4_multimodal_best.pt --split val
```

This now reports both:
- `Oracle` metrics: best of the 3 predicted modes
- `Top-1` metrics: only the highest-confidence mode

## Export Predictions

```bash
python3 predict_phase4.py --checkpoint checkpoints/phase4_multimodal_best.pt --split val --output outputs/phase4_predictions_val.json
```

## Visualize Predictions

```bash
python3 visualize_phase4.py --checkpoint checkpoints/phase4_multimodal_best.pt --split val --num-samples 5
```

## Generate Report Assets

```bash
python3 report_phase4.py --checkpoint checkpoints/phase4_multimodal_best.pt --split val --output-dir outputs/phase4_report
```

## Files

- `trajectory_baseline/dataset.py`: nuScenes pedestrian dataset
- `trajectory_baseline/model.py`: LSTM baseline
- `train_phase1.py`: training and validation loop
- `train_phase2.py`: Transformer + map encoder training loop
- `train_phase3.py`: Transformer + map + social encoder training loop
- `train_phase4.py`: multimodal trajectory, probability, and intent training loop
- `evaluate_phase4.py`: reports minADE, minFDE, Miss Rate, intent accuracy, confusion matrix
- `predict_phase4.py`: exports predictions to JSON
- `visualize_phase4.py`: saves PNG visualizations of map + history + GT + predicted modes
- `report_phase4.py`: saves metrics JSON plus presentation-ready plots

## Notes

- The loader uses a deterministic scene-level train/val split on the mini set.
- Tracks are converted into the local frame of the last observed position.
- Phase 2 adds `100x100` grayscale local map patches from nuScenes semantic prior maps.
- Phase 3 adds nearby pedestrian histories within a `10m` radius and pads them to a fixed neighbor count.
- Phase 4 adds heuristic intent labels and predicts `K=3` trajectories with mode probabilities.
- Phase 5 adds demo- and submission-friendly tooling around the Phase 4 checkpoint.
- Intent labels are heuristic and dataset-dependent; the mini validation split is heavily straight-walking dominated.
- The map raster alignment is a lightweight approximation so it stays runnable without `nuscenes-devkit`.
