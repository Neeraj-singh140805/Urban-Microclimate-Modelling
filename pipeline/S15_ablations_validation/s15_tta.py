"""
S15 test-time ablation: the S11 tuned networks without averaging over exact symmetric copies.

Prediction normally averages each case over its exact copies (the mirror, and at 45 degrees also the
transpose). This rebuilds fold 0's predictions of the three S11 tuned seeds from their saved weights
with that averaging switched off, so the comparison changes nothing but the prediction step. Writes
results/preds/s15_no_tta_seed<s>/ (not tracked); s15_results.py scores them against the same seed.

Usage: uv run python -m pipeline.S15_ablations_validation.s15_tta
"""

from __future__ import annotations

from urbanwind.dl.train import (
    CHECKPOINTS,
    PREDS,
    REPO_ROOT,
    fold_table,
    load_best,
    load_config,
    pick_device,
    write_predictions,
)

CONFIG = REPO_ROOT / "pipeline" / "S11_unet_baseline" / "configs" / "unet_tuned.yaml"


def main() -> None:
    cfg = load_config(CONFIG) | {"mirror_tta": False}
    device = pick_device()
    dev = fold_table("grouped")
    cases = dev.index[dev["fold"] == 0].tolist()
    for seed in range(3):
        model = load_best(cfg, CHECKPOINTS / f"s11_unet_tuned_seed{seed}" / "fold0.pt", device)
        write_predictions(model, cases, PREDS / f"s15_no_tta_seed{seed}", device, cfg)
        print(f"seed {seed}: {len(cases)} fold-0 cases predicted without symmetric averaging", flush=True)


if __name__ == "__main__":
    main()
