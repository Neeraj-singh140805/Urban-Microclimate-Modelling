"""The README banner: the simulated pedestrian-level wind of one development case, with the project title.

Needs the S10 cache (pipeline/S10_dl_data_pipeline/build_cache.py).
Usage: uv run python reports/figures/src/make_banner.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from urbanwind.dl.cache import load_cached  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "banner.jpg"
CASE, ROW = "AU-Preston", 300  # a development case, never a test case
WIND = LinearSegmentedColormap.from_list("wind", ["#061226", "#0b2f5e", "#1a56a0", "#3583dc", "#82b2ec", "#e3eefc"])
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def main() -> None:
    c = load_cached(CASE)
    U, H = c["U"].astype(float), c["H"].astype(float)
    width = U.shape[1]
    u, building = U[ROW : ROW + width // 4, :width], (H > 0)[ROW : ROW + width // 4, :width]
    rgb = WIND(np.clip(np.nan_to_num(u, nan=0) / np.nanpercentile(u, 99.5), 0, 1))[..., :3]
    rgb[building] = (0.06, 0.065, 0.075)
    fig = plt.figure(figsize=(12.8, 3.2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgb, aspect="auto", interpolation="bilinear", extent=[0, 1, 0, 1])
    x = np.linspace(0, 1, 1024)  # dark on the left, where the text sits, fading to the right
    alpha = np.clip(0.93 - 0.55 * np.clip((x - 0.45) / 0.35, 0, 1) - 0.38 * np.clip((x - 0.8) / 0.2, 0, 1), 0, 1)
    shade = np.zeros((1, x.size, 4))
    shade[..., :3], shade[..., 3] = (0.025, 0.04, 0.075), alpha
    ax.imshow(shade, aspect="auto", extent=[0, 1, 0, 1])
    text = {"family": FONT, "va": "center"}
    ax.text(0.04, 0.62, "Urban Microclimate Modelling", color="white", fontsize=40, fontweight="bold", **text)
    ax.text(
        0.04,
        0.42,
        "Neural surrogates of large-eddy simulation for pedestrian-level wind in unseen cities",
        color="#d6e5f8",
        fontsize=15,
        **text,
    )
    ax.text(
        0.04,
        0.24,
        "538 UrbanTALES simulations  ·  grouped cross-validation  ·  periodic U-Net",
        color="#a9c0de",
        fontsize=12,
        **text,
    )
    ax.text(
        0.988,
        0.045,
        f"Simulated wind 1.75 m above the ground, UrbanTALES case {CASE} (Nazarian et al., 2025), CC BY-NC 4.0",
        color="#e6eef9",
        fontsize=8.5,
        family=FONT,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#050a14", "ec": "none", "alpha": 0.75},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    png = OUT.with_suffix(".png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    Image.open(png).convert("RGB").save(OUT, quality=90, optimize=True)
    png.unlink()


if __name__ == "__main__":
    main()
