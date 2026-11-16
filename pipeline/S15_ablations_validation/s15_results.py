"""
S15: each ablation against the S11 tuned U-Net on fold 0's 97 cases.

Every ablation changes one setting of pipeline/S11_unet_baseline/configs/unet_tuned.yaml and trains
fold 0 with seed 0 (configs/). Its predictions are rebuilt from the saved weights with
`python -m urbanwind.dl.train --config <config> --seed 0 --folds 0 --repredict`.

Two yardsticks:
- the spread of the three S11 tuned seeds on the same cases: a change smaller than that spread is
  not distinguishable from a different training seed;
- a paired comparison with S11 seed 0 (for prediction-only variants, the seed they reuse), case by case, with 95% intervals from 2,000 resamples of
  fold 0's city and layout groups. With seed 0 on both sides and the same training cases, the patches
  are drawn in the same order, so the setting is the only planned difference.

The prediction-only variants of s15_tta.py (the S11 weights, predicted without symmetric averaging)
are compared with the S11 seed whose weights they reuse.

Runs repeated on folds 1-4 as well are compared the same way on all development cases, where the
paired comparison also counts the folds in which the ablation is better.

For those runs it also lists the cases that add the most squared error against S11 seed 0, and how
large the gap in pooled R² is without them, to tell a broad loss from one driven by a few cases.

Writes outputs/fold0_ablations.csv, outputs/fold0_vs_s11.csv and, once some run covers all five folds,
outputs/all_folds_ablations.csv, outputs/all_folds_vs_s11.csv and outputs/all_folds_worst_cases.csv.

Usage: uv run python -m pipeline.S15_ablations_validation.s15_results
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from pipeline.S11_unet_baseline.s11_results import bootstrap_vs_s07, pooled_r2, score_run
from urbanwind.dl.train import PREDS, fold_table

STAGE = pathlib.Path(__file__).resolve().parent
OUT, RUNS = STAGE / "outputs", STAGE / "outputs" / "runs"
S11_RUNS = STAGE.parent / "S11_unet_baseline" / "outputs" / "runs"
S11_SEEDS = [f"s11_unet_tuned_seed{s}" for s in range(3)]
REF = S11_SEEDS[0]
NO_TTA = {f"s15_no_tta_seed{s}": f"s11_unet_tuned_seed{s}" for s in range(3)}  # prediction-only variants


def finished() -> list[str]:
    """S15 runs whose fold 0 has finished and whose predictions exist, then the prediction-only variants."""
    trained = sorted(p.parent.name for p in RUNS.glob("*/fold0.json") if (PREDS / p.parent.name).is_dir())
    return trained + [run for run in NO_TTA if (PREDS / run).is_dir()]


def reference(run: str) -> str:
    return NO_TTA.get(run, REF)


def record(run: str) -> dict:
    """The fold-0 training record; a prediction-only variant shares the record of the weights it reuses."""
    name = NO_TTA.get(run, run)
    return json.loads(((S11_RUNS if name in S11_SEEDS else RUNS) / name / "fold0.json").read_text())


def metrics(run: str, sub: pd.DataFrame) -> dict:
    by = {s: sub[sub["subset"] == s] for s in ("idealized", "realistic")}
    return {
        "run": run,
        "n_cases": len(sub),
        "pooled_r2": pooled_r2(sub),
        "median_case_r2": sub["r2"].median(),
        "pooled_r2_idealized": pooled_r2(by["idealized"]),
        "pooled_r2_realistic": pooled_r2(by["realistic"]),
        "median_case_r2_idealized": by["idealized"]["r2"].median(),
        "median_case_r2_realistic": by["realistic"]["r2"].median(),
        "mean_mae": sub["mae"].mean(),
        "mean_delta_p95": sub["delta_p95"].mean(),
    }


def row(run: str, sub: pd.DataFrame, record: dict) -> dict:
    return metrics(run, sub) | {
        "n_training_cases": record.get("n_fit"),
        "inner_val_rmse": record.get("best_val"),
        "best_epoch": record.get("best_epoch"),
        "train_minutes": record.get("minutes"),
    }


def against_seeds(table: pd.DataFrame) -> pd.DataFrame:
    """Marks each run above, within or below the range of the three S11 seeds (MAE: lower is better)."""
    s11 = table[table["run"].isin(S11_SEEDS)]
    for metric in ("pooled_r2", "median_case_r2", "pooled_r2_realistic", "mean_mae"):
        lo, hi = s11[metric].min(), s11[metric].max()
        table[f"{metric}_vs_s11_seeds"] = [
            "" if r in S11_SEEDS else ("above" if v > hi else "below" if v < lo else "within")
            for r, v in zip(table["run"], table[metric], strict=True)
        ]
    return table


def mean_prediction(run: str, case: str) -> float:
    with np.load(PREDS / run / f"{case}.npz") as f:
        return float(f["y_pred"][~f["mask"]].mean())


def worst_cases(scored: pd.DataFrame, run: str, k: int = 5) -> pd.DataFrame:
    """The k cases that add the most squared error against S11 seed 0, and the pooled-R² gap left without them."""
    a, b = (scored[scored["run"] == r].set_index("case") for r in (run, REF))
    extra = a["sse"] - b["sse"]
    top = extra.sort_values(ascending=False).index[:k]
    rows = []
    for i, case in enumerate(top):
        rest = ~a.index.isin(top[: i + 1])
        rows.append(
            {
                "run": run,
                "rank": i + 1,
                "case": case,
                "outer_fold": a.loc[case, "outer_fold"],
                "subset": a.loc[case, "subset"],
                "share_of_cells": a.loc[case, "n"] / a["n"].sum(),
                "share_of_extra_sse": extra[case] / extra.sum(),
                "mean_true": a.loc[case, "mean_true"],
                "mean_pred": mean_prediction(run, case),
                "mean_pred_ref": mean_prediction(REF, case),
                "r2": a.loc[case, "r2"],
                "r2_ref": b.loc[case, "r2"],
                "delta_pooled_r2_without_cases_so_far": pooled_r2(a[rest]) - pooled_r2(b[rest]),
            }
        )
    return pd.DataFrame(rows)


def all_folds(scored: pd.DataFrame, dev: pd.DataFrame) -> None:
    """Runs predicted on every development case, against the S11 seeds on all five folds."""
    covered = scored.groupby("run")["case"].nunique()
    full = [run for run in covered.index[covered == len(dev)] if run not in S11_SEEDS]
    if not full:
        return
    runs = [*S11_SEEDS, *sorted(full)]
    table = against_seeds(pd.DataFrame([metrics(run, scored[scored["run"] == run]) for run in runs]))
    paired = pd.concat([bootstrap_vs_s07(scored, run, ref=REF) for run in sorted(full)], ignore_index=True)
    table.to_csv(OUT / "all_folds_ablations.csv", index=False, float_format="%.4f")
    paired.to_csv(OUT / "all_folds_vs_s11.csv", index=False, float_format="%.4f")
    worst = pd.concat([worst_cases(scored, run) for run in sorted(full)], ignore_index=True)
    worst.to_csv(OUT / "all_folds_worst_cases.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.precision", 4, "display.max_columns", 30):
        print("\nall five folds\n", table.to_string(index=False), "\n")
        print(paired[paired["subset"] == "all"].to_string(index=False))


def main() -> None:
    dev = fold_table("grouped")
    fold0 = dev.index[dev["fold"] == 0]
    runs = [*S11_SEEDS, *finished()]
    every = pd.concat([score_run(run, dev) for run in runs], ignore_index=True)
    scored = every[every["case"].isin(fold0)]
    table = against_seeds(pd.DataFrame([row(run, scored[scored["run"] == run], record(run)) for run in runs]))
    if not finished():
        raise SystemExit("no finished S15 run has predictions yet; rebuild them with --repredict")
    paired = pd.concat(
        [bootstrap_vs_s07(scored, run, ref=reference(run)).assign(ref_run=reference(run)) for run in finished()],
        ignore_index=True,
    ).drop(columns="folds_better")  # one fold: folds_better says nothing
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "fold0_ablations.csv", index=False, float_format="%.4f")
    paired.to_csv(OUT / "fold0_vs_s11.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.precision", 4, "display.max_columns", 30):
        print(table.to_string(index=False), "\n")
        print(paired[paired["subset"] == "all"].to_string(index=False))
    all_folds(every, dev)


if __name__ == "__main__":
    main()
