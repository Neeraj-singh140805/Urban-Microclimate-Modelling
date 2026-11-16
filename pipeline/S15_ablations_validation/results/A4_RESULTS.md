# A4 — Loss Function Comparison

## Experiment scope
- Outer fold: 0
- Seed: 0
- Epochs: 100 per experiment
- Predictions: 97 per experiment
- Status: Fold 0 completed; five-fold validation pending.

## Metrics
See `a4_fold0_seed0_metrics.csv` for the reconstructed metrics.

## Current observation
FFT currently leads with pooled R² of 0.7911, followed by
Sobel at 0.7868.

This is a preliminary Fold 0 result, not a final loss selection.
Further fold validation is required.

## Important recovery note
The metrics were reconstructed from the recovered experiment
results. Original checkpoint and prediction binaries were not
recovered. The CSV is a metrics record, not a replacement for
those original artifacts.

The idealized and realistic columns contain subset pooled R².
They should not be interpreted as median per-case R² by subset.
