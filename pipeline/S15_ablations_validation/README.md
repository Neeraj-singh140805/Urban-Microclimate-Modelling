# S15 Ablations and technical validation

Asks which parts of the S11 tuned U-Net recipe matter: each ablation removes or changes one part and measures what that costs. It also records how fast the final model predicts a case.

| | |
| --- | --- |
| Inputs | the S10 cache, the S11 tuned predictions and weights (`results/preds/` and `results/checkpoints/s11_unet_tuned_seed{0,1,2}/`, not tracked), the weights of the final S13 recipe (`results/checkpoints/s13_unet_tuned_e200_seed0/`, not tracked) |
| Outputs | `outputs/fold0_ablations.csv`, `outputs/fold0_vs_s11.csv`, `outputs/all_folds_ablations.csv`, `outputs/all_folds_vs_s11.csv`, `outputs/all_folds_worst_cases.csv`, `outputs/inference_time.csv`; `outputs/runs/<exp_id>/fold<k>.json` and `.csv` |
| Code | `configs/` (one file per ablation), `s15_results.py`, `s15_tta.py`, `s15_speed.py`; the options `padding`, `channels`, `train_subset` and `train_group_fraction` in `src/urbanwind/dl/nets.py` and `train.py`; tests in `src/urbanwind/dl/tests/test_s15_options.py` |

Status: complete for development data. 19 ablations ran on fold 0, six of them on all five folds, and test-time mirror averaging was ablated for all three S11 seeds. The test regimes are scored once, with the final model, in S16.

## 1. Method

### One change at a time

Every config in `configs/` is `pipeline/S11_unet_baseline/configs/unet_tuned.yaml` (100 epochs) with one setting changed. Two ablations switch off a pair of settings together, as noted in the tables. The options that did not exist before S15 leave the settings hash of earlier runs unchanged at their default values, so the S11 to S14 records still match their configs; a test checks this.

| Option | What it does |
| --- | --- |
| `model.kwargs.padding: zeros` | every convolution pads with zeros instead of wrapping around, and upsampling no longer wraps either |
| `channels: [...]` | the network sees only the listed S10 input channels (`height`, `building`, `wall_distance`, `cos_angle`, `sin_angle`); the dropped channels are removed before the first layer, so the network is the same size apart from its first convolution |
| `train_subset: idealized` or `realistic` | trains only on cases of that subset; every development case is still scored |
| `train_group_fraction: f` | trains on a random share `f` of the training split groups (cities or layout families), whole groups only and the same share of each subset, drawn with a fixed seed per fold |

### Two yardsticks

Each ablation trains with seed 0. Where its training cases are the same as S11's, it draws its patches in the same order as S11 seed 0, so the change is the only planned difference.

1. **The spread of the three S11 seeds.** A run is `above`, `within` or `below` the range of the three S11 tuned seeds on the same cases. A change inside that range cannot be told apart from a different training seed.
2. **A paired comparison with S11 seed 0**, case by case, with 95% intervals from 2,000 resamples of whole split groups (`bootstrap_vs_s07` from S11 with S11 seed 0 as the reference). On all five folds it also counts the folds in which the ablation is better.

### Which ablations ran on all five folds

Fold 0 (97 cases) screened all 19. Six were repeated on folds 1 to 4, one or two for each claim the network design rests on: height-only input (inputs), zero padding (periodic domain), no augmentation (symmetry), tail-weighted loss (peaks), half the cities (data) and idealized-only training (transfer to real cities).

## 2. Results on fold 0 (`outputs/fold0_ablations.csv`, `outputs/fold0_vs_s11.csv`)

The three S11 seeds on fold 0: pooled R² 0.7887 to 0.8020, median per-case R² 0.567 to 0.633, idealized pooled R² 0.830 to 0.849, realistic pooled R² 0.784 to 0.797, mean MAE 0.442 to 0.465 u*.

ΔR² is against S11 seed 0 (pooled R² 0.7907), with its 95% interval. "Seeds" places the pooled R² against the S11 seed range.

| Factor | Ablation | Training cases | Pooled R² | ΔR² [95% interval] | Seeds | Idealized pooled R² | Realistic pooled R² | Mean ΔP95 (u*) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | S11 tuned, seed 0 | 324 | 0.791 |  |  | 0.838 | 0.786 | -0.486 |
| A1 inputs | height only | 324 | 0.757 | -0.033 [-0.172, -0.021] | below | 0.533 | 0.765 | -0.602 |
|  | height and building mask | 324 | 0.758 | -0.032 [-0.163, -0.021] | below | 0.534 | 0.766 | -0.621 |
| A1, A3 inputs | height, mask and wall distance (no wind direction) | 324 | 0.766 | -0.024 [-0.152, -0.013] | below | 0.550 | 0.773 | -0.625 |
| A2 padding | zeros instead of circular | 324 | 0.745 | -0.045 [-0.106, -0.039] | below | 0.703 | 0.744 | -0.359 |
| A4 loss | MSE | 324 | 0.765 | -0.026 [-0.029, -0.010] | below | 0.820 | 0.759 | -0.527 |
|  | L1 plus Sobel gradient | 324 | 0.757 | -0.033 [-0.037, -0.015] | below | 0.828 | 0.751 | -0.520 |
|  | tail-weighted L1, γ = 1 | 324 | 0.740 | -0.051 [-0.066, -0.049] | below | 0.782 | 0.734 | -0.351 |
|  | tail-weighted L1, γ = 2 | 324 | 0.606 | -0.185 [-0.309, -0.167] | below | 0.643 | 0.599 | -0.256 |
|  | L1 plus FFT magnitude | 324 | 0.790 | -0.000 [-0.011, 0.001] | within | 0.834 | 0.785 | -0.463 |
|  | L1 plus mean consistency | 324 | 0.750 | -0.041 [-0.086, -0.035] | below | 0.748 | 0.746 | -0.583 |
| A6 data | 25% of the city and layout groups | 95 | 0.629 | -0.162 [-0.188, -0.025] | below | 0.737 | 0.619 | -0.414 |
|  | 50% of the groups | 201 | 0.764 | -0.026 [-0.054, -0.023] | below | 0.778 | 0.761 | -0.486 |
|  | 75% of the groups | 269 | 0.770 | -0.021 [-0.040, -0.018] | below | 0.791 | 0.766 | -0.485 |
|  | idealized cases only | 128 | 0.041 | -0.749 [-0.793, -0.407] | below | 0.905 | -0.013 | -0.286 |
|  | realistic cases only | 193 | 0.779 | -0.012 [-0.103, -0.001] | below | 0.602 | 0.784 | -0.581 |
| A7 augmentation | no mirror and no 45° transpose | 324 | 0.703 | -0.088 [-0.103, -0.009] | below | 0.751 | 0.696 | -0.593 |
| A8 regularisation | no weight decay and no dropout | 324 | 0.791 | +0.000 [-0.001, 0.007] | within | 0.842 | 0.786 | -0.484 |
|  | no weight averaging (EMA) | 324 | 0.787 | -0.003 [-0.004, 0.000] | below | 0.840 | 0.782 | -0.491 |
|  | no early stopping: all training cases, final weights | 385 | 0.772 | -0.019 [-0.021, 0.001] | below | 0.839 | 0.766 | -0.471 |

Test-time mirror averaging, ablated by predicting with the S11 weights without it (`s15_tta.py`), compared with the seed whose weights it reuses:

| S11 seed | Pooled R² with averaging | Without | ΔR² [95% interval] |
| --- | --- | --- | --- |
| 0 | 0.7907 | 0.7856 | -0.005 [-0.010, -0.004] |
| 1 | 0.7887 | 0.7831 | -0.006 [-0.008, -0.005] |
| 2 | 0.8020 | 0.7973 | -0.005 [-0.007, -0.004] |

## 3. Results on all five folds (`outputs/all_folds_ablations.csv`, `outputs/all_folds_vs_s11.csv`)

All 482 development cases. The three S11 seeds: pooled R² 0.7092 to 0.7122, median per-case R² 0.560 to 0.595, idealized pooled R² 0.935 to 0.944, realistic pooled R² 0.677 to 0.680, mean ΔP95 -0.457 to -0.439 u*.

| Ablation | Pooled R² | ΔR² [95% interval] | Folds better | Cases better | Idealized pooled R² | Realistic pooled R² | Mean ΔP95 (u*) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S11 tuned, seed 0 | 0.709 |  |  |  | 0.935 | 0.677 | -0.457 |
| height only | 0.665 | -0.044 [-0.086, -0.015] | 0 of 5 | 13% | 0.644 | 0.667 | -0.600 |
| zero padding | 0.678 | -0.031 [-0.042, -0.023] | 0 of 5 | 29% | 0.851 | 0.653 | -0.354 |
| tail-weighted L1, γ = 1 | 0.675 | -0.034 [-0.061, -0.009] | 1 of 5 | 12% | 0.909 | 0.641 | -0.335 |
| 50% of the groups | 0.528 | -0.181 [-0.364, -0.026] | 0 of 5 | 21% | 0.912 | 0.473 | -0.482 |
| no augmentation | 0.679 | -0.030 [-0.057, -0.011] | 0 of 5 | 39% | 0.836 | 0.656 | -0.555 |
| idealized cases only | 0.090 | -0.619 [-0.708, -0.489] | 0 of 5 | 39% | 0.963 | -0.035 | -0.302 |

"Cases better" is the share of cases with a higher per-case R² than S11 seed 0. `outputs/all_folds_worst_cases.csv` lists, for each run, the five cases that add the most squared error and the gap in pooled R² that remains without them.

## 4. What the ablations show

- **Every part of the input is used, and the wind direction most of all on idealized cases.** On fold 0, height alone loses 0.033 pooled R². Adding the building mask changes almost nothing (+0.001), as the mask follows from the height. Adding the wall distance recovers 0.008, and the wind direction the remaining 0.024. Without the direction channels, idealized pooled R² falls from 0.838 to 0.550: the idealized arrays are simulated at 0° and 45° on the same layout, and inside the canonical frame only the direction channels tell those two cases apart. On all five folds, height alone loses 0.044, with idealized pooled R² falling from 0.935 to 0.644.
- **Circular padding matters.** With zero padding the network sees each domain edge as a border of zeros instead of the continuation of the periodic domain. It loses 0.031 pooled R² over five folds, in every fold, with the largest cost on idealized cases (0.935 to 0.851). Its predicted 95th percentile is closer to the truth (mean ΔP95 -0.354 u* against -0.457), so the loss is not from flattening the peaks.
- **Plain L1 is the best of the seven losses tried.** Only the FFT term ties it (within the seed range on every measure); every other loss is worse with an interval that excludes zero. Tail weighting does what it is meant to, lifting the predicted peaks (mean ΔP95 -0.335 u* against -0.457 over five folds), but costs 0.034 pooled R²; with γ = 2 the cost on fold 0 is 0.185.
- **Augmentation by the exact symmetries is worth about 0.03.** Without the mirror and 45° transpose of the training patches, pooled R² falls by 0.030 over five folds, in every fold, and idealized pooled R² from 0.935 to 0.836. About half of the extra squared error comes from one fast, open suburb, AU-Preston (true mean 5.5 u*, predicted 3.7 against 5.4 by S11 seed 0); without it the loss is 0.017. Averaging the mirrored predictions at test time adds about 0.005 more, for each of the three seeds, with intervals that exclude zero.
- **More cities help, and the gain has not levelled off.** On fold 0, pooled R² rises from 0.629 with a quarter of the groups to 0.765, 0.770 and 0.791 with half, three quarters and all of them. Over five folds, half the groups loses 0.181 pooled R², but most of that comes from two cases in fold 3. Trained on half of fold 3's groups (146 cases instead of 325), the network predicted mean speeds of 1.8 and 2.0 u* for US-WestPhoenix and CA-Sunset, two open suburbs whose true means are 6.1 u* (S11 seed 0: 5.3 and 4.6). These two cases hold 1.8% of the scored cells and 87% of the run's extra squared error; without them, half the groups loses 0.030 pooled R². A smaller training set leaves gaps in the range of layouts, and a case that falls into such a gap can be predicted very badly. The fastest, most open suburbs are the most exposed: AU-Preston also carries the largest single loss of no augmentation, and US-WestPhoenix, AU-Preston and CA-Sunset the three largest of idealized-only training.
- **Idealized arrays do not teach the network real cities, and real cities cost it a little on idealized arrays.** Trained on idealized cases only, the network scores a realistic pooled R² of -0.035 over five folds: worse than predicting one overall mean speed for every cell. Trained on realistic cases only, it matches the full network on realistic cases on fold 0 (0.784, within the seed range) and loses on idealized ones (0.602 against 0.838). The idealized-only network is also better than the full one on idealized cases, by 0.029 pooled R² [0.022, 0.042], in all five folds. Mixing in realistic cases therefore costs some accuracy on idealized arrays.
- **The regularisation settings matter little.** Without weight decay and dropout, or without the weight average, the result stays within or just below the seed range, with intervals that include zero. Training on all 385 cases with the final weights, instead of holding out 61 cases to pick the best epoch, gives 0.772: below the seed range, although the interval [-0.021, 0.001] includes zero. The 61 extra training cases did not help measurably.

## 5. Planned factors, and what was not run

| Factor | Where it is answered |
| --- | --- |
| A1 input channels | here: height, plus mask, plus wall distance, plus direction. Shelter and fetch maps were not tried as network inputs; the S13 residual model (H3) combined the network's prediction with the S07 per-cell features, which include upwind fetch and shelter, and did not improve pooled R² |
| A2 padding | here |
| A3 direction | here, the canonical frame with and without the direction channels. The alternative with D4 augmentation and a direction input was not run |
| A4 loss | here |
| A5 architecture | S12 (Residual U-Net, Attention U-Net, FNO) and S13 (ConvNeXt U-Net, longer training) |
| A6 data | here, including idealized-only and realistic-only training |
| A7 augmentation | here, for training and for test-time averaging |
| A8 regularisation | here |
| A9 hybrid parts | S13 (H1 level test, H3 residual stacking) |
| A10 resolution | every score above: all networks train at 1 m, and the idealized cases, stored at 0.5 m, are scored on their native 0.5 m grid after periodic bilinear upsampling |

Not run, with the reason:

- **Three seeds per ablation.** Each ablation has one seed. The spread of the three S11 seeds stands in for the seed noise, and the paired interval against the same seed measures the change itself. Where a result is close to the seed range, the text says so.
- **Leave-one-country-out.** The development folds already hold out whole cities; holding out whole countries needs its own splits and trainings.
- **Shelter and fetch inputs, D4 with a direction input.** Both need new input code in S10; see the table above.

The test regimes (unseen cities, λp = 0.35, λp = 0.64) are scored once with the final model in S16. The idealized-to-realistic regime is answered above on development data.

## 6. Inference time (`s15_speed.py`, `outputs/inference_time.csv`)

The final recipe (`pipeline/S13_hybrid/configs/unet_tuned_e200.yaml`, the seed 0, fold 0 weights) predicts each of fold 0's 97 cases as it does for scoring, with mirror averaging and on the native grid. Measured on an Apple M3 Pro GPU (PyTorch MPS):

| | Cases | Median | Largest |
| --- | --- | --- | --- |
| Idealized | 38 | 0.10 s | 0.18 s |
| Realistic | 59 | 0.23 s | 0.94 s |

The final model averages three seeds, so it takes about three times as long: under a second for most cases and about 3 s for the largest. Two runs back to back agreed within 11 ms on the medians; an earlier run on the same machine gave medians of 0.075 s and 0.19 s, so the times depend on the state of the machine.

## Reproducibility

Each run trained on a Kaggle T4 GPU, about 65 minutes per fold (zero padding about 40), from a code bundle at a recorded commit with no local changes (`train_code` in each `fold<k>.json`). Kaggle stores the best weights; the predictions were then rebuilt from those weights on the local machine and scored:

```bash
# predictions from saved weights (no training), then the official evaluator once all five folds exist
uv run python -m urbanwind.dl.train --config pipeline/S15_ablations_validation/configs/<name>.yaml --seed 0 --folds 0 --repredict
uv run python -m pipeline.S15_ablations_validation.s15_tta      # S11 weights without test-time averaging
uv run python -m pipeline.S15_ablations_validation.s15_results  # tables in outputs/
uv run python -m pipeline.S15_ablations_validation.s15_speed    # inference time
```

To train a run from scratch, drop `--repredict`; each fold takes about an hour on a T4.
