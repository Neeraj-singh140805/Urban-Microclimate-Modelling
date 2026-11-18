"""
S16: scoring of the frozen test set with the final Phase 2 model.

Every modelling choice was fixed on the development folds (S11 to S15) before this script was
written. The final model is the 200-epoch U-Net of S13, trained with three seeds on each of the
five development folds. Every one of those 15 networks is out of sample for a test case, so the
prediction is their cell-by-cell average, made from the building raster and the wind direction
only (urbanwind.dl.predict). The 53 cases of the 24 frozen test groups are scored once, with the
same per-case metrics, pooling and subsets as the Phase 1 test scoring (S09).

Two test cases come from cities that also have development neighbourhoods (AU-SURREYHILLS in
Melbourne, FI-MS in Helsinki), so every score is also reported without them.

The script refuses to run again once outputs/test_scored.json exists, so the test set cannot be
used to make choices. The guard is written before anything is scored. A second scoring is
possible only with --rescore and a reason; the guard then keeps the full scoring history.

Writes results/preds/s16_unet_final/<case>.npz (evaluator format, not tracked) and
outputs/test_field_per_case.csv, outputs/test_field_level.csv, outputs/test_vs_phase1.csv.

Usage: uv run python pipeline/S16_frozen_test/s16_score.py [--rescore "reason"]   (a few minutes)
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd
import torch

from urbanwind.data import EXCLUDE, REPO_ROOT, case_table, load_case
from urbanwind.dl.datasets import predict_arrays
from urbanwind.dl.predict import FINAL_CONFIG, checkpoint_paths, layout_arrays
from urbanwind.dl.train import inference, load_best, load_config, pick_device
from urbanwind.metrics import case_metrics
from urbanwind.splits import load_splits

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
GUARD = OUT / "test_scored.json"
PREDS = REPO_ROOT / "results" / "preds" / "s16_unet_final"
S09 = REPO_ROOT / "pipeline" / "S09_phase1_evaluation" / "outputs"
SHARED_CITY = {"city-AU-SURREYHILLS", "city-FI-MS"}  # as in S09
SEEDS, FOLDS = (0, 1, 2), range(5)


def test_cases() -> pd.DataFrame:
    table = case_table()
    test = table[table["split_group"].isin(load_splits()["test_groups"]) & ~table.index.isin(EXCLUDE)]
    return test.assign(shared_city=test["split_group"].isin(SHARED_CITY))[["subset", "split_group", "shared_city"]]


def load_networks(cfg: dict, device: torch.device) -> list[torch.nn.Module]:
    return [load_best(cfg, p, device).eval() for p in checkpoint_paths(cfg, SEEDS, FOLDS)]


def predict_case(c: dict, nets: list[torch.nn.Module], cfg: dict, device: torch.device) -> np.ndarray:
    """Average of the networks on the native grid, from the building raster and the wind direction only."""
    arrays = layout_arrays(c["H"], c["dx"], c["wind_dir_deg"])
    autocast = torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type != "cpu")
    with torch.no_grad(), autocast:
        preds = [predict_arrays(net, arrays, device, mirror_tta=cfg["mirror_tta"], upsample=cfg["native_upsample"],
                                **inference(cfg)) for net in nets]
    return np.mean(preds, axis=0)


def score(test: pd.DataFrame, nets: list[torch.nn.Module], cfg: dict, device: torch.device) -> pd.DataFrame:
    PREDS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, row in test.iterrows():
        c = load_case(case)
        truth = c["U"].astype(np.float32)
        valid = np.isfinite(truth)
        pred = predict_case(c, nets, cfg, device)
        np.savez_compressed(PREDS / f"{case}.npz", y_true=truth, y_pred=np.where(valid, pred, np.nan).astype(np.float32),
                            mask=~valid)
        yc, pc = truth[valid].astype(float), pred[valid].astype(float)
        rows.append({"case": case, "subset": row["subset"], "shared_city": row["shared_city"], "n_cells": len(yc),
                     "sum_y": yc.sum(), "sum_y2": (yc**2).sum(), "sse": ((yc - pc) ** 2).sum(),
                     **{k: v for k, v in case_metrics(yc, pc).items() if k != "case"}})
        print(f"{case}: R2 {rows[-1]['r2']:.3f}", flush=True)
    return pd.DataFrame(rows)


def summary(per_case: pd.DataFrame) -> pd.DataFrame:
    """Same pooling and subsets as S09's field_summary."""
    rows = []
    for label, sub in (("all", per_case), ("idealized", per_case[per_case["subset"] == "idealized"]),
                       ("realistic", per_case[per_case["subset"] == "realistic"]),
                       ("all without shared-city cases", per_case[~per_case["shared_city"]])):
        n, s, ss, sse = sub[["n_cells", "sum_y", "sum_y2", "sse"]].sum()
        rows.append({"variant": "unet_final", "cases": label, "n_cases": len(sub), "pooled_r2": 1 - sse / (ss - s * s / n),
                     "median_within_case_r2": sub["r2"].median(), "mean_mae": sub["mae"].mean(),
                     "mean_delta_p95": sub["delta_p95"].mean(), "mean_level_err": sub["level_err"].mean(),
                     "mean_pattern_err": sub["pattern_err"].mean()})
    return pd.DataFrame(rows)


def versus_phase1(per_case: pd.DataFrame) -> pd.DataFrame:
    """Per-case R2 of the U-Net against the Phase 1 pixel model on the same test cases."""
    p1 = pd.read_csv(S09 / "test_field_per_case.csv").query("variant == 'pixel_gbm'").set_index("case")["r2"]
    df = per_case.set_index("case")[["subset", "r2", "mae"]].join(p1.rename("r2_phase1_pixel"), how="inner")
    return df.assign(r2_gain=df["r2"] - df["r2_phase1_pixel"]).reset_index()


def write_guard(history: list[dict], status: str, **info) -> None:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout
    entry = {"date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "git_sha": sha.strip(),
             "status": status, **info}
    GUARD.write_text(json.dumps({**entry, "history": history}, indent=1))


def main(rescore: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if GUARD.exists():
        previous = json.loads(GUARD.read_text())
        if not rescore:
            raise SystemExit(f"the frozen test set was already scored on {previous['date']}; see {GUARD}")
        history = [*previous.pop("history"), previous]
    reason = {"reason": rescore} if rescore else {}
    test = test_cases()
    cfg = load_config(FINAL_CONFIG)
    device = pick_device()
    nets = load_networks(cfg, device)  # fails before the guard if any weight file is missing
    write_guard(history, "started", n_networks=len(nets), config=str(FINAL_CONFIG.relative_to(REPO_ROOT)), **reason)
    per_case = score(test, nets, cfg, device)
    per_case.drop(columns=["n_cells", "sum_y", "sum_y2", "sse"]).to_csv(OUT / "test_field_per_case.csv", index=False,
                                                                        float_format="%.6f")
    summary(per_case).to_csv(OUT / "test_field_level.csv", index=False, float_format="%.6f")
    versus_phase1(per_case).to_csv(OUT / "test_vs_phase1.csv", index=False, float_format="%.6f")
    write_guard(history, "scored", n_test_cases=len(test), n_networks=len(nets),
                config=str(FINAL_CONFIG.relative_to(REPO_ROOT)), **reason)
    print(summary(per_case).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rescore", help="reason for scoring the test set a second time")
    main(parser.parse_args().rescore)
