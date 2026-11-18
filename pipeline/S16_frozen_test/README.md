# S16 Frozen test set

Scores the final Phase 2 model once on the 53 cases of the 24 frozen test groups, after every modelling choice was fixed on the development folds (S11 to S15).

| | |
| --- | --- |
| Inputs | the raw UrbanTALES files of the test cases through the shared loader; the weights of the final S13 recipe (`results/checkpoints/s13_unet_tuned_e200_seed{0,1,2}/fold{0..4}.pt`, not tracked); the Phase 1 test scores (`pipeline/S09_phase1_evaluation/outputs/`) |
| Outputs | `outputs/test_field_level.csv`, `outputs/test_field_per_case.csv`, `outputs/test_vs_phase1.csv`, `outputs/test_scored.json`; predictions in `results/preds/s16_unet_final/` (not tracked); one row in `results/runs.csv` |
| Code | `s16_score.py`, using `urbanwind.dl.predict` and `urbanwind.metrics` |

Status: scored on 18 November 2026.

## Method

- **Final model.** The 200-epoch U-Net of S13 (`pipeline/S13_hybrid/configs/unet_tuned_e200.yaml`), trained with three seeds on each of the five development folds. All 15 networks are out of sample for a test case, so the prediction is their cell-by-cell average. It is made from the building raster and the wind direction only, with the inference settings of the development scores (whole-domain pass, mirror averaging, bilinear upsampling of 0.5 m grids).
- **Metrics.** The per-case metrics, pooling and subsets of the Phase 1 test scoring (S09), so the two phases are compared on the same cases in the same way. The shared evaluator (`python -m urbanwind.evaluate results/preds/s16_unet_final --final-test`) gives the same pooled and median per-case $R^2$.
- **Once only.** The script writes `outputs/test_scored.json` before it scores anything and refuses to run again unless given `--rescore` and a reason. The tables were written a second time on the same day with six decimals instead of four, so that the report's three-decimal values are not rounded twice; the networks and predictions were unchanged (largest per-case difference 0.00005, the old rounding). No test score was used for any choice.

## Results

| Cases | n | Model | Pooled $R^2$ | Median per-case $R^2$ | MAE ($u_*$) | Mean $\Delta P_{95}$ ($u_*$) |
| --- | --- | --- | --- | --- | --- | --- |
| all | 53 | Final U-Net | **0.840** | **0.655** | **0.362** | **-0.353** |
| | | Phase 1 pixel model | 0.680 | 0.351 | 0.530 | -0.651 |
| idealized | 34 | Final U-Net | **0.960** | **0.699** | **0.256** | **-0.220** |
| | | Phase 1 pixel model | 0.893 | 0.425 | 0.389 | -0.481 |
| realistic | 19 | Final U-Net | **0.805** | **0.620** | **0.552** | **-0.589** |
| | | Phase 1 pixel model | 0.617 | 0.289 | 0.783 | -0.956 |
| all without shared-city cases | 51 | Final U-Net | 0.795 | 0.655 | 0.352 | -0.349 |

- **Better on every case.** The U-Net has the higher per-case $R^2$ in all 53 test cases, idealized and realistic (`outputs/test_vs_phase1.csv`).
- **Higher than in development for both models.** In development the pooled $R^2$ is 0.734 for the U-Net and 0.591 for the pixel model. The test set is 64% idealized against 39% in development, and idealized cases are easier for both.
- **Hardest cases.** The lowest per-case $R^2$ are US53_d45 (idealized, 0.114; Phase 1 -1.146) and the two US-New-V1 directions (realistic, 0.123 and 0.226).
- **Shared cities.** Two test cases come from cities that also have development neighbourhoods (AU-SURREYHILLS in Melbourne, FI-MS in Helsinki); without them the pooled $R^2$ is 0.795.

Reproduce (refuses to run again; see above): `uv run python pipeline/S16_frozen_test/s16_score.py`
