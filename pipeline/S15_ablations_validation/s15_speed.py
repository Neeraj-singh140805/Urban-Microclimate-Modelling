"""
S15: inference time of the final model, one whole case at a time.

Loads the weights of the final recipe (pipeline/S13_hybrid/configs/unet_tuned_e200.yaml, seed 0, fold 0)
and predicts each of fold 0's 97 cases with the settings used for scoring (mirror averaging, native
grid). The clock starts after the case is read from the cache, so disk time is not counted. The first
three cases are predicted once more beforehand to warm the device up. The final model averages three
seeds, so it takes about three times the time measured here.

Writes outputs/inference_time.csv (seconds per case, with the device and chip).

Usage: uv run python -m pipeline.S15_ablations_validation.s15_speed
"""

from __future__ import annotations

import pathlib
import platform
import subprocess
import time

import pandas as pd
import torch

from urbanwind.dl.cache import load_cached
from urbanwind.dl.datasets import predict_arrays
from urbanwind.dl.train import CHECKPOINTS, REPO_ROOT, fold_table, inference, load_best, load_config, pick_device

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
FINAL = REPO_ROOT / "pipeline/S13_hybrid/configs/unet_tuned_e200.yaml"
N_WARM = 3


def chip(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    return platform.processor() or "CPU"


def main() -> None:
    cfg = load_config(FINAL)
    device = pick_device()
    net = load_best(cfg, CHECKPOINTS / f"{cfg['exp_id']}_seed0" / "fold0.pt", device).eval()
    cases = [c for c, f in fold_table(cfg["split"])["fold"].items() if f == 0]
    amp = torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type != "cpu")
    rows = []
    with torch.no_grad(), amp:
        for i, case in enumerate([*cases[:N_WARM], *cases]):
            c = load_cached(case)
            t0 = time.perf_counter()
            predict_arrays(
                net, c, device=device, mirror_tta=cfg["mirror_tta"], upsample=cfg["native_upsample"], **inference(cfg)
            )
            if i >= N_WARM:
                rows.append(
                    {
                        "case": case,
                        "subset": c["subset"],
                        "cells_1m": int(c["H"].size),
                        "seconds": time.perf_counter() - t0,
                        "device": str(device),
                        "chip": chip(device),
                        "torch": torch.__version__,
                    }
                )
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "inference_time.csv", index=False, float_format="%.4f")
    print(table.groupby("subset")["seconds"].describe().round(4))


if __name__ == "__main__":
    main()
