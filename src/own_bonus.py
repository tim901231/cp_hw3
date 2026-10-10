"""Bonus 4: reflection removal (Agrawal et al. Sec. 4.1, Observation Deck).

Run from the repository root:
    python src/own_bonus.py

The pair was taken on a tripod and is used without alignment.
The flash lights the figure in front of the window but also creates reflections of
room objects on the glass; the ambient (long exposure) image shows the outside scene
without reflections.
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import report_figures_p2 as R
from main import (readimg, save_jpg, divergence, gradient_cgd, reflection_free_gradient,
                  linear_rgb_to_luminance)

AMBIENT = "data/own/window_ambient.jpg"
FLASH = "data/own/window_flash.jpg"
OUT_DIR = "output/bonus_reflection"  # figures
RESULT_DIR = "results/bonus_reflection"
WIDTH = 1500
SIGMA, TAU_UE = 40, 0.1
EPS, N_ITER = 1e-3, 5000

R.FIG_DIR = OUT_DIR
R.CROPS = {"Reflection": (150, 330, 540, 360), "Figure": (530, 420, 540, 360)}


def load(path):
    img = readimg(path)
    h = round(img.shape[0] * WIDTH / img.shape[1])
    return cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA).astype(np.float64)


def solve(fields, boundary):
    out = np.zeros_like(boundary)
    for c in range(3):
        out[..., c], h = gradient_cgd(divergence(fields[c]), boundary[..., c], boundary[..., c], EPS, N_ITER)
        print(f"  channel {c}: {len(h) - 1} iterations")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = load(AMBIENT)
    f = load(FLASH)

    lum = linear_rgb_to_luminance(a)     # normalized (display-referred) ambient intensity
    fields, refl, info = [], [], None
    for c in range(3):
        g, q = reflection_free_gradient(a[..., c], f[..., c], lum, SIGMA, TAU_UE)
        fields.append(g)
        refl.append(q["g_reflection"])
        info = q
    H = a + f
    print("integrating reflection-free image")
    out = solve(fields, H)
    print("integrating reflection layer")
    layer = solve(refl, np.zeros_like(H))

    save_jpg(os.path.join(RESULT_DIR, "window_reflection_removed.jpg"), out)
    save_jpg(os.path.join(RESULT_DIR, "window_H.jpg"), H)
    save_jpg(os.path.join(RESULT_DIR, "window_reflection_layer.jpg"), np.abs(layer) * 3)
    np.savez(os.path.join(OUT_DIR, "result.npz"), out=out, layer=layer, H=H, a=a, f=f, w_ue=info["w_ue"])
    make_figures(a, f, H, out, layer, info["w_ue"])


def make_figures(a, f, H, out, layer, w_ue):
    R.fig_inputs(a, f)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.2), gridspec_kw=dict(wspace=0.03, hspace=0.14))
    panels = [(a, "Ambient $a$"), (f, r"Flash $\Phi'$"), (H, r"$H=a+\Phi'$"),
              (None, r"Underexposed map $w_{ue}$"), (np.clip(np.abs(layer) * 3, 0, 1), r"Reflection layer ($3\times$)"),
              (out, r"Result $\Phi^*$")]
    for ax, (img, t) in zip(axes.ravel(), panels):
        if img is None:
            im = ax.imshow(w_ue, cmap="magma", vmin=0, vmax=1)
            R.clean_ax(ax)
            ax.set_title(t, pad=3)
        else:
            R.show(ax, img, t)
    R.save(fig, "bonus_overview.png")

    cols = [(a, "Ambient $a$"), (f, r"Flash $\Phi'$"), (H, r"$H$"), (out, r"Result $\Phi^*$")]
    fig, axes = plt.subplots(len(R.CROPS), len(cols), figsize=(7.0, 2.55),
                             gridspec_kw=dict(wspace=0.03, hspace=0.04))
    for i, (name, color) in enumerate(zip(R.CROPS, R.CROP_COLORS)):
        for j, (img, t) in enumerate(cols):
            R.show(axes[i, j], R.crop(img, name), t if i == 0 else None)
            R.frame(axes[i, j], color)
    R.save(fig, "bonus_crops.png")
    fig_gradients(a, f, w_ue)


def fig_gradients(a, f, w_ue, crop="Reflection"):
    """x and y gradient components of the ambient image, H and the fused field,
    and the alignment cos^2 between grad H and grad a (fraction kept by the projection)."""
    lum = linear_rgb_to_luminance(a)
    g_a, g_H, g_s = [], [], []
    for c in range(3):
        g, q = reflection_free_gradient(a[..., c], f[..., c], lum, SIGMA, TAU_UE)
        g_a.append(q["g_a"]); g_H.append(q["g_H"]); g_s.append(g)
    g_a, g_H, g_s = (np.stack(v, axis=-1) for v in (g_a, g_H, g_s))      # (2, H, W, 3)
    names = [(g_a, r"$\nabla a$"), (g_H, r"$\nabla H$"), (g_s, r"$\nabla\Phi^*$")]

    x, y, w, h = R.CROPS[crop]
    sl = (slice(y, y + h), slice(x, x + w))
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 7.0 / 4 * h / w * 2 + 0.35),
                             gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for j, (g, t) in enumerate(names):
        R.show_signed(axes[0, j], g[1][sl], 0.05, t + r": $x$")
        R.show_signed(axes[1, j], g[0][sl], 0.05, t + r": $y$")
    # cos^2 of the angle between grad H and grad a (green channel)
    dot = np.sum(g_H[..., 1] * g_a[..., 1], axis=0)
    cos2 = dot ** 2 / (np.sum(g_H[..., 1] ** 2, axis=0) * np.sum(g_a[..., 1] ** 2, axis=0) + 1e-12)
    ax = axes[0, 3]
    im = ax.imshow(cos2[sl], cmap="magma", vmin=0, vmax=1)
    R.clean_ax(ax)
    ax.set_title(r"$\cos^2\angle(\nabla H,\nabla a)$", pad=3)
    ax = axes[1, 3]
    ax.imshow(w_ue[sl], cmap="magma", vmin=0, vmax=1)
    R.clean_ax(ax)
    ax.set_title(r"$w_{ue}$", pad=3)
    cax = axes[0, 3].inset_axes([1.03, 0, 0.05, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2)
    R.save(fig, "bonus_gradients.png")


if __name__ == "__main__":
    main()
