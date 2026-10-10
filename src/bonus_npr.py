"""Bonus 5: gradient-domain non-photorealistic rendering (GradientShop-style).

Run from the repository root:
    python src/bonus_npr.py

For each image:
  1. Long-edge detection: OpenCV Canny on the bilateral-filtered grayscale image,
     keeping only connected edge chains longer than MIN_LENGTH pixels
     (instead of the GradientShop long-edge detector).
  2. Gradient editing: g = (ALPHA * e + BETA * (1 - e)) * grad(u), which strengthens
     long edges and flattens texture and shading.
  3. Integration of div(g) with the CGD Poisson solver (Dirichlet boundary from u).
"""
import os
import sys

import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from main import readimg, save_jpg, grad, divergence, gradient_cgd, bilateral_filtering

OUT_DIR = "output/bonus_npr"
WIDTH = 1000
SWEEP_WIDTH = 600                 # smaller images for the parameter sweep
ALPHA, BETA = 2.0, 0.3            # gain on long edges / elsewhere
MIN_LENGTH = 80                   # minimum edge-chain length in pixels
EPS, N_ITER = 1e-3, 3000

EXAMPLES = {
    "photo1": "data/own/npr_1.jpg",
    "photo2": "data/own/npr_2.jpg",
    "photo3": "data/own/npr_3.jpg",
}
SWEEP_IMAGE = "data/museum/museum_flash.png"   # parameter sweep (Figure in the report)
RESULT_DIR = "results/bonus_npr"

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def load(path, width=WIDTH):
    # Apply the EXIF orientation (phone photos are stored rotated)
    img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), dtype=np.float32) / 255.0
    h = round(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA).astype(np.float64)


def long_edge_map(img, min_length=MIN_LENGTH, sigma_s=3, sigma_r=0.1, blur=1.5):
    """Canny edges on the bilateral-filtered image, keeping only long connected chains.
    Returns the binary Canny map, the long-edge map and a smooth weight e in [0, 1]."""
    smooth = bilateral_filtering(img.astype(np.float32), img.astype(np.float32),
                                 sigma_s=sigma_s, sigma_r=sigma_r)
    gray = (np.clip(cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY), 0, 1) * 255).astype(np.uint8)
    med = np.median(gray)
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))

    # Keep connected edge chains (8-connectivity) that are long enough
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (stats[:, cv2.CC_STAT_AREA] >= min_length) | (extent >= min_length)
    keep[0] = False
    long_edges = keep[labels]

    e = cv2.dilate(long_edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float64)
    e = np.clip(cv2.GaussianBlur(e, (0, 0), blur), 0, 1)
    e = e / max(e.max(), 1e-8)
    return edges > 0, long_edges, e


def npr(img, alpha=ALPHA, beta=BETA, **edge_kw):
    canny, long_edges, e = long_edge_map(img, **edge_kw)
    scale = alpha * e + beta * (1 - e)
    out = np.zeros_like(img)
    for c in range(3):
        g = scale * grad(img[..., c])
        out[..., c], _ = gradient_cgd(divergence(g), img[..., c], img[..., c], EPS, N_ITER)
    return out, dict(canny=canny, long_edges=long_edges, e=e)


def show(ax, img, title=None, **kw):
    ax.imshow(np.clip(img, 0, 1), **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, pad=3)


def fig_example(name, img, out, info):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 7.0 / 4 * img.shape[0] / img.shape[1] + 0.3),
                             gridspec_kw=dict(wspace=0.03))
    show(axes[0], img, "Input")
    show(axes[1], 1 - info["canny"], "Canny edges", cmap="gray")
    show(axes[2], 1 - info["long_edges"], f"Long edges ($\\geq${MIN_LENGTH} px)", cmap="gray")
    show(axes[3], out, "NPR result")
    path = os.path.join(OUT_DIR, f"npr_{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def fig_sweep(name, img):
    """min_length is scaled with the image width so the edge selection matches."""
    betas, alphas = [0.6, 0.3, 0.1], [1.0, 2.0, 3.0]
    fig, axes = plt.subplots(len(betas), len(alphas),
                             figsize=(7.0, 7.0 / 3 * img.shape[0] / img.shape[1] * 3 + 0.3),
                             gridspec_kw=dict(wspace=0.03, hspace=0.03))
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            out, _ = npr(img, alpha=a, beta=b, min_length=round(MIN_LENGTH * img.shape[1] / WIDTH))
            show(axes[i, j], out, rf"$\alpha={a:g}$" if i == 0 else None)
            if j == 0:
                axes[i, j].set_ylabel(rf"$\beta={b:g}$")
    path = os.path.join(OUT_DIR, f"npr_sweep_{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def main(names=None):
    """Run the given examples (all by default); the sweep runs only when all are run."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, path in EXAMPLES.items():
        if names and name not in names:
            continue
        img = load(path)
        out, info = npr(img)
        save_jpg(os.path.join(RESULT_DIR, f"{name}_npr.jpg"), out)
        fig_example(name, img, out, info)
    if not names:
        fig_sweep("museum", load(SWEEP_IMAGE, SWEEP_WIDTH))


if __name__ == "__main__":
    main(sys.argv[1:])
