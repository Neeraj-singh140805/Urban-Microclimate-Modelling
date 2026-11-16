"""S15 ablation options: zero padding, input-channel selection and restricted training cases."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from urbanwind.data import REPO_ROOT
from urbanwind.dl.nets import UNet
from urbanwind.dl.train import DEFAULTS, config_hash, load_config, make_model, training_cases

SMALL = {"widths": (8, 16, 16, 16), "bottleneck": 16, "dropout": 0.0}


def cfg_with(**changes) -> dict:
    return DEFAULTS | {"model": {"name": "unet", "kwargs": SMALL}} | changes


@pytest.mark.parametrize(
    ("config", "record"),
    [
        (
            "pipeline/S11_unet_baseline/configs/unet_tuned.yaml",
            "pipeline/S11_unet_baseline/outputs/runs/s11_unet_tuned_seed0/fold0.json",
        ),
        (
            "pipeline/S13_hybrid/configs/unet_tuned_e200.yaml",
            "pipeline/S13_hybrid/outputs/runs/s13_unet_tuned_e200_seed0/fold0.json",
        ),
    ],
)
def test_existing_runs_keep_their_settings_hash(config, record):
    recorded = json.loads((REPO_ROOT / record).read_text())["config_hash"]
    assert config_hash(load_config(REPO_ROOT / config)) == recorded


def test_ablation_settings_change_the_hash():
    base = cfg_with()
    assert config_hash(cfg_with(channels=["height"])) != config_hash(base)
    assert config_hash(cfg_with(train_subset="idealized")) != config_hash(base)
    assert config_hash(cfg_with(train_group_fraction=0.5)) != config_hash(base)


def test_zero_padding_has_the_same_weights_but_is_not_periodic():
    torch.manual_seed(0)
    circ = UNet(in_ch=5, **SMALL).double().eval()
    zero = UNet(in_ch=5, padding="zeros", **SMALL).double().eval()
    zero.load_state_dict(circ.state_dict())  # same parameters, only the edge handling differs
    x = torch.randn(1, 5, 64, 64, dtype=torch.float64)
    shifted = torch.roll(x, shifts=(16, 16), dims=(-2, -1))
    with torch.no_grad():
        assert torch.allclose(circ(shifted), torch.roll(circ(x), (16, 16), (-2, -1)), atol=1e-10)
        assert not torch.allclose(zero(shifted), torch.roll(zero(x), (16, 16), (-2, -1)), atol=1e-6)
    with pytest.raises(ValueError):
        UNet(in_ch=5, padding="reflect", **SMALL)


def test_channel_selection_feeds_only_the_kept_channels():
    torch.manual_seed(0)
    model = make_model(cfg_with(channels=["height", "wall_distance"])).eval()
    x = torch.randn(2, 5, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), model.net(x[:, [0, 2]]))
    assert model.multiple == 16
    clone = make_model(cfg_with(channels=["height", "wall_distance"]))
    clone.load_state_dict(model.state_dict())  # weights saved and reloaded through make_model
    with pytest.raises(ValueError):
        make_model(cfg_with(channels=["roughness"]))


def test_default_model_is_the_plain_network():
    keys = set(make_model(cfg_with()).state_dict())
    assert keys == set(UNet(in_ch=5, **SMALL).state_dict())


def test_training_cases_subset_and_group_share():
    rows = [(f"{s}{g}_{i}", s, f"{s}-g{g}") for s in ("idealized", "realistic") for g in range(10) for i in range(3)]
    train = pd.DataFrame(rows, columns=["case", "subset", "split_group"]).set_index("case")
    only = training_cases(train, cfg_with(train_subset="idealized"), fold=0)
    assert set(only["subset"]) == {"idealized"} and len(only) == 30
    half = training_cases(train, cfg_with(train_group_fraction=0.5), fold=0)
    assert half.groupby("subset")["split_group"].nunique().to_dict() == {"idealized": 5, "realistic": 5}
    assert half.groupby("split_group").size().eq(3).all()  # whole groups only
    assert half.index.equals(training_cases(train, cfg_with(train_group_fraction=0.5), fold=0).index)
    assert not half.index.equals(training_cases(train, cfg_with(train_group_fraction=0.5), fold=1).index)
    assert training_cases(train, cfg_with(), fold=0).equals(train)
