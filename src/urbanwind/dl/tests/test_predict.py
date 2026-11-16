"""Prediction for a new layout: orientation, grid spacing, ensembling and building cells."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from urbanwind.dl.datasets import predict_arrays
from urbanwind.dl.nets import UNet
from urbanwind.dl.predict import layout_arrays, predict_layout

SMALL = {"widths": [8, 16, 16, 16], "bottleneck": 16, "dropout": 0.0}


def layout(ny: int = 64, nx: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = np.zeros((ny, nx))
    for _ in range(6):
        r, c = rng.integers(0, ny - 8), rng.integers(0, nx - 8)
        H[r : r + 6, c : c + 6] = rng.uniform(5, 30)
    return H


def small_net(seed: int) -> UNet:
    torch.manual_seed(seed)
    return UNet(in_ch=5, **SMALL).eval()


def test_turning_the_layout_and_the_wind_turns_the_prediction():
    """np.rot90 turns the wind by -90 degrees, so both layouts reach the same wind-aligned frame."""
    H, net = layout(), small_net(0)
    with torch.no_grad():
        a = predict_arrays(net, layout_arrays(H, 1.0, 120.0), "cpu", mirror_tta=True)
        b = predict_arrays(net, layout_arrays(np.rot90(H), 1.0, 30.0), "cpu", mirror_tta=True)
    assert a.shape == H.shape and b.shape == np.rot90(H).shape
    assert np.allclose(np.rot90(a), b, atol=1e-5)


def test_a_half_metre_raster_gets_a_half_metre_prediction():
    H = np.kron(layout(32, 24), np.ones((2, 2)))  # 0.5 m cells
    with torch.no_grad():
        y = predict_arrays(small_net(0), layout_arrays(H, 0.5, 10.0), "cpu")
    assert y.shape == H.shape


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        layout_arrays(np.full((8, 8), np.nan), 1.0, 0.0)
    with pytest.raises(ValueError):
        layout_arrays(layout(), 2.0, 0.0)


def test_predict_layout_averages_the_networks_and_blanks_buildings(tmp_path):
    cfg = {"exp_id": "tiny", "model": {"name": "unet", "kwargs": SMALL}}
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(cfg))
    for seed in (0, 1):
        (tmp_path / f"tiny_seed{seed}").mkdir()
        torch.save(small_net(seed).state_dict(), tmp_path / f"tiny_seed{seed}" / "fold0.pt")
    H = layout()
    out = predict_layout(
        H, 1.0, 30.0, config, seeds=(0, 1), folds=(0,), checkpoint_dir=tmp_path, device=torch.device("cpu")
    )
    assert out["n_networks"] == 2 and out["U"].shape == H.shape
    assert np.isnan(out["U"][H > 0]).all() and np.isfinite(out["U"][H == 0]).all()
    assert (out["spread"][H == 0] > 0).any()
    with pytest.raises(FileNotFoundError):
        predict_layout(H, 1.0, 30.0, config, seeds=(5,), folds=(0,), checkpoint_dir=tmp_path)
