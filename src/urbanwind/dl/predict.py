"""
Pedestrian-level wind for a new building layout, from the trained networks.

Input: a building-height raster in metres (0 on open ground) on a 1 m or 0.5 m grid, with rows along +y and
columns along +x, of a neighbourhood that repeats periodically, and the direction the wind blows towards, in
degrees from +x (columns) towards +y (rows). Nothing from a simulation is needed.

Output: the time-averaged wind speed at 1.75 m divided by the friction velocity, U/u*, on the input grid,
averaged over the given networks, with their standard deviation as an uncertainty map. Building cells are NaN.

Usage:
  uv run python -m urbanwind.dl.predict --height layout.npy --wind-dir 30 --out wind.npz --png wind.png
  uv run python -m urbanwind.dl.predict --case AU-Ade-V1 --out wind.npz   # an UrbanTALES case from data/raw

The weights are not in the repository; see "Using the trained model" in the README.
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Iterable

import numpy as np
import torch

from urbanwind.dl.cache import FINE_DX, canonical_transform, make_inputs, resample_to_1m, to_canonical
from urbanwind.dl.datasets import predict_arrays
from urbanwind.dl.train import CHECKPOINTS, REPO_ROOT, inference, load_best, load_config, pick_device
from urbanwind.features import compute_periodic_distance_transform

FINAL_CONFIG = REPO_ROOT / "pipeline" / "S13_hybrid" / "configs" / "unet_tuned_e200.yaml"


def layout_arrays(H: np.ndarray, dx: float, wind_dir_deg: float) -> dict:
    """What ``predict_arrays`` needs for one layout: the raster in the wind-aligned frame and its inputs."""
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or not np.isfinite(H).all() or (H < 0).any():
        raise ValueError("the height raster must be a 2D array of finite, non-negative heights in metres")
    if not (np.isclose(dx, 1.0) or np.isclose(dx, FINE_DX)):
        raise ValueError(f"the grid spacing must be 1 m or 0.5 m, not {dx}")
    H1, _ = resample_to_1m(H, {}, dx)
    k, transposed, angle = canonical_transform(wind_dir_deg)
    Hc = to_canonical(H1, k, transposed)
    dist, _ = compute_periodic_distance_transform(Hc, dx=1.0)
    return {
        "H": Hc,
        "dist": dist,
        "angle_deg": angle,
        "k": k,
        "transposed": transposed,
        "native_shape": np.array(H.shape),
        "dx_native": float(dx),
        "x": make_inputs(Hc, dist, angle),
    }


def checkpoint_paths(
    cfg: dict, seeds: Iterable[int], folds: Iterable[int], checkpoint_dir: pathlib.Path = CHECKPOINTS
) -> list[pathlib.Path]:
    paths = [checkpoint_dir / f"{cfg['exp_id']}_seed{s}" / f"fold{k}.pt" for s in seeds for k in folds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing weights: {missing}; see 'Using the trained model' in the README")
    return paths


def predict_layout(
    H: np.ndarray,
    dx: float,
    wind_dir_deg: float,
    config: pathlib.Path = FINAL_CONFIG,
    seeds: Iterable[int] = (0, 1, 2),
    folds: Iterable[int] = range(5),
    checkpoint_dir: pathlib.Path = CHECKPOINTS,
    device: torch.device | None = None,
) -> dict:
    """U/u* on the input grid: the mean over the networks, their standard deviation, and the building mask.

    Each fold's networks were trained on four fifths of the development cases, so for a layout outside
    the dataset all of them are out of sample, and averaging them is an ensemble. Prediction uses the
    settings of the reported scores (mirror averaging, whole-domain pass, bilinear upsampling of 0.5 m grids).
    """
    cfg = load_config(pathlib.Path(config))
    device = device or pick_device()
    arrays = layout_arrays(H, dx, wind_dir_deg)
    preds = []
    autocast = torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type != "cpu")
    for path in checkpoint_paths(cfg, seeds, folds, checkpoint_dir):
        net = load_best(cfg, path, device).eval()
        with torch.no_grad(), autocast:
            preds.append(
                predict_arrays(
                    net, arrays, device, mirror_tta=cfg["mirror_tta"], upsample=cfg["native_upsample"], **inference(cfg)
                )
            )
    stack = np.stack(preds)
    building = np.asarray(H) > 0
    mean, spread = stack.mean(axis=0), stack.std(axis=0)
    mean[building], spread[building] = np.nan, np.nan
    return {
        "U": mean.astype(np.float32),
        "spread": spread.astype(np.float32),
        "building": building,
        "n_networks": len(preds),
        "wind_dir_deg": float(wind_dir_deg),
        "dx": float(dx),
    }


def save_png(result: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt  # only needed for the quick-look figure

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, key, cmap, label in ((axes[0], "U", "Blues", "U / u*"), (axes[1], "spread", "Oranges", "spread (u*)")):
        im = ax.imshow(result[key], cmap=cmap, origin="lower")
        ax.imshow(np.where(result["building"], 0.75, np.nan), cmap="Greys", vmin=0, vmax=1, origin="lower")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    axes[0].set_title(f"wind towards {result['wind_dir_deg']:g} deg (from +x towards +y)")
    axes[1].set_title(f"spread of {result['n_networks']} networks")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def read_height(path: pathlib.Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as f:
            return f["H"]
    if path.suffix in (".csv", ".txt"):
        return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    raise ValueError(f"{path}: use .npy, .npz (array 'H'), .csv or .txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pedestrian-level wind (U/u*) for a building layout.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--height", type=pathlib.Path, help="height raster in metres: .npy, .npz (H), .csv or .txt")
    source.add_argument("--case", help="an UrbanTALES case name, read from data/raw with its own wind direction")
    parser.add_argument("--wind-dir", type=float, help="direction the wind blows towards, degrees from +x to +y")
    parser.add_argument("--dx", type=float, default=1.0, help="grid spacing of the raster in metres (1 or 0.5)")
    parser.add_argument("--config", type=pathlib.Path, default=FINAL_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ustar", type=float, help="friction velocity in m/s; adds U in m/s to the output")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="output .npz")
    parser.add_argument("--png", type=pathlib.Path, help="optional quick-look figure")
    args = parser.parse_args(argv)
    if args.case:
        from urbanwind.data import load_case

        c = load_case(args.case)
        H, dx, wind = c["H"], c["dx"], c["wind_dir_deg"]
    else:
        if args.wind_dir is None:
            parser.error("--wind-dir is required with --height")
        H, dx, wind = read_height(args.height), args.dx, args.wind_dir
    result = predict_layout(H, dx, wind, args.config, args.seeds, args.folds)
    if args.ustar:
        result["U_ms"] = result["U"] * np.float32(args.ustar)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    if args.png:
        save_png(result, args.png)
    open_cells = ~result["building"]
    print(
        f"{result['n_networks']} networks; mean U/u* over open ground {np.nanmean(result['U']):.3f}; "
        f"mean spread {np.nanmean(result['spread']):.3f}; {open_cells.sum()} open cells -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
