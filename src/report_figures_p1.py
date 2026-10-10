"""Generate Problem 1 report figures (lamp flash/no-flash pair).

Run from the repository root:
    python src/report_figures_p1.py

Outputs PNG figures to report/figures/ and caches filter results in
output/cache/ so that re-styling figures does not re-run the filters.
Cache files are keyed by (sigma_s, sigma_r); other parameters are cheap and
recomputed on every run.
"""
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
from main import (bilateral_filtering, brute_force_bilateral, readimg, to_linear,
                  linear_rgb_to_luminance, clean_mask, shadow_specular_mask)

# ---------------------------------------------------------------- config ----
FIG_DIR = "report/figures"
CACHE_DIR = "output/cache"

# Parameters of the final pipeline; keep in sync with main.p1().
CHOSEN = {
    "base": dict(sigma_s=16, sigma_r=0.05),    # A_Base: piecewise, ambient guides ambient
    "flash": dict(sigma_s=16, sigma_r=0.05),   # F_Base: piecewise, flash guides flash
    "nr": dict(sigma_s=4, sigma_r=0.001),      # A_NR:   brute force, flash guides ambient
    "eps": 0.02,                               # detail transfer epsilon
    "iso_ambient": 1600, "iso_flash": 200,
    "tau_shadow": 0.0,                         # Y_F - Y_A*ISO_F/ISO_A <= tau
    "spec_thresh": 0.95,                       # Y_F > 0.95 (95% of sensor range)
    "open_k": 3, "close_k": 5, "dilate_k": 3,  # clean_mask, applied to each mask
    "feather_sigma": 2,
}

# Parameter sweeps (rows: sigma_r, columns: sigma_s).
SWEEP_SIGMA_S = [4, 16, 64]
SWEEP_SIGMA_R = [0.05, 0.10, 0.25]
SWEEP_EPS = [0.002, 0.02, 0.1]

# Zoom regions (x, y, w, h) in full-resolution pixels.
CROPS = {
    "Lamp strands": (1000, 1150, 360, 360),
    "Pan on wall": (2080, 780, 360, 360),
}
CROP_COLORS = ["#ffcc33", "#33ccff"]
# Display-only brightness gain per crop (the pan region is very dark).
CROP_GAIN = {"Lamp strands": 1.0, "Pan on wall": 2.5}
DIFF_LIM = 0.08   # signed per-channel difference shown as 0.5 + d / (2 * DIFF_LIM)

# ------------------------------------------------------------------ style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
DIFF_CMAP = "RdBu_r"


# ---------------------------------------------------------------- helpers ----
def crop(img, key):
    x, y, w, h = CROPS[key]
    return img[y:y + h, x:x + w]


def dcrop(img, key):
    """Crop for display, with the per-crop brightness gain applied."""
    return crop(img, key) * CROP_GAIN[key]


def small(img, f=0.25):
    return cv2.resize(img.astype(np.float32), None, fx=f, fy=f,
                      interpolation=cv2.INTER_AREA)


def lum(img):
    """Linear luminance (used for the shadow/specularity mask)."""
    return linear_rgb_to_luminance(to_linear(img))


def luma(img):
    """Display-referred luma Y' of the non-linear image (used for difference maps)."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def frame(ax, color):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(color)
        s.set_linewidth(2.0)


def show(ax, img, title=None, **kw):
    ax.imshow(np.clip(img, 0, 1), interpolation="lanczos", **kw)
    clean_ax(ax)
    if title:
        ax.set_title(title, pad=3)


def show_diff(ax, d, title=None):
    """Signed RGB difference: mid-gray = 0, each channel mapped linearly to +/-DIFF_LIM."""
    ax.imshow(np.clip(0.5 + d / (2 * DIFF_LIM), 0, 1), interpolation="nearest")
    clean_ax(ax)
    if title:
        ax.set_title(title, pad=3)
    return matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(-DIFF_LIM, DIFF_LIM), cmap="gray")


DIFF_LABEL = "signed difference per RGB channel (gray = 0)"


def inset_colorbar(fig, ax, im, label=None, extend="both"):
    cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0])
    cb = fig.colorbar(im, cax=cax, extend=extend)
    cb.ax.tick_params(labelsize=7, length=2)
    if label:
        cb.set_label(label, fontsize=8)
    return cb


def draw_crop_boxes(ax, scale):
    for (x, y, w, h), c in zip(CROPS.values(), CROP_COLORS):
        ax.add_patch(Rectangle((x * scale, y * scale), w * scale, h * scale,
                               fill=False, ec=c, lw=1.4))


def overlay(img, masks_colors, alpha=0.55):
    out = np.clip(img, 0, 1).copy()
    for m, c in masks_colors:
        out[m] = (1 - alpha) * out[m] + alpha * np.array(matplotlib.colors.to_rgb(c))
    return out


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


# ------------------------------------------------------------------ cache ----
def bf_tag(prefix, p):
    return f"{prefix}_s{p['sigma_s']}_r{p['sigma_r']:.3f}"


def cached(name, fn):
    path = os.path.join(CACHE_DIR, name + ".npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    out = fn().astype(np.float32)
    np.save(path, out.astype(np.float16))
    return out


# --------------------------------------------------------------- pipeline ----
def compute_mask(A, F, c=CHOSEN):
    """Mask components for the figures; M itself comes from main.shadow_specular_mask."""
    ya, yf = lum(A), lum(F)
    shadow_raw = yf - ya * c["iso_flash"] / c["iso_ambient"] <= c["tau_shadow"]
    spec_raw = yf > c["spec_thresh"]
    morph = dict(open_k=c["open_k"], close_k=c["close_k"], dilate_k=c["dilate_k"])
    M = shadow_specular_mask(A, F, iso_ambient=c["iso_ambient"], iso_flash=c["iso_flash"],
                             tau_shadow=c["tau_shadow"], spec_thresh=c["spec_thresh"],
                             feather_sigma=c["feather_sigma"])
    return dict(shadow_raw=shadow_raw, shadow=clean_mask(shadow_raw, **morph),
                spec_raw=spec_raw, spec=clean_mask(spec_raw, **morph), M=M)


def run_pipeline(A, F):
    c = CHOSEN
    A_base = cached(bf_tag("Abase", c["base"]),
                    lambda: bilateral_filtering(A, A, **c["base"]))
    F_base = cached(bf_tag("Fbase", c["flash"]),
                    lambda: bilateral_filtering(F, F, **c["flash"]))
    A_nr = cached(bf_tag("Anr_brute", c["nr"]),
                  lambda: brute_force_bilateral(A, F, **c["nr"]))
    A_detail = A_nr * (F + c["eps"]) / (F_base + c["eps"])
    masks = compute_mask(A, F)
    M = masks["M"][..., None]
    A_final = (1 - M) * A_detail + M * A_base
    return dict(A_base=A_base, F_base=F_base, A_nr=A_nr, A_detail=A_detail,
                A_final=A_final, **masks)


STAGES = [("A", "Ambient $A$"), ("A_base", r"$A^{Base}$"), ("A_nr", r"$A^{NR}$"),
          ("A_detail", r"$A^{Detail}$"), ("A_final", r"$A^{Final}$")]


# ---------------------------------------------------------------- figures ----
def fig_inputs(A, F):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.03))
    for ax, img, t in zip(axes, [A, F],
                          [f"Ambient $A$ (ISO {CHOSEN['iso_ambient']})",
                           f"Flash $F$ (ISO {CHOSEN['iso_flash']})"]):
        show(ax, small(img), t)
        draw_crop_boxes(ax, 0.25)
    handles = [Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=1.4) for c in CROP_COLORS]
    fig.legend(handles, list(CROPS), loc="lower center", ncol=len(CROPS),
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    save(fig, "p1_inputs.png")


def sweep_results(A, guide, prefix):
    res = {}
    for sr in SWEEP_SIGMA_R:
        for ss in SWEEP_SIGMA_S:
            p = dict(sigma_s=ss, sigma_r=sr)
            res[(sr, ss)] = cached(bf_tag(prefix, p),
                                   lambda p=p: bilateral_filtering(A, guide, **p))
    return res


def _sweep_figure(A, res, fname, title, residual, ref=None, ref_tex="$A$"):
    """Two blocks (one per crop): [input | gap | sigma_s sweep]; rows are sigma_r."""
    nr, ns = len(SWEEP_SIGMA_R), len(SWEEP_SIGMA_S)
    W = 7.0
    cell = W / (2 * (ns + 1.25) + 0.3)
    fig = plt.figure(figsize=(W, cell * nr * 1.05 + 0.5))
    outer = fig.add_gridspec(1, 2, wspace=0.06, left=0, right=1, bottom=0, top=0.88)
    im = None
    for b, (name, color) in enumerate(zip(CROPS, CROP_COLORS)):
        g = outer[b].subgridspec(nr, ns + 2, wspace=0.03, hspace=0.03,
                                 width_ratios=[1, 0.25] + [1] * ns)
        ax = fig.add_subplot(g[0, 0])
        show(ax, dcrop(ref if residual else A, name), ref_tex if residual else "input $A$")
        frame(ax, color)
        ax.set_xlabel(name, fontsize=7.5, color="0.3", labelpad=2)
        for i, sr in enumerate(SWEEP_SIGMA_R):
            for j, ss in enumerate(SWEEP_SIGMA_S):
                ax = fig.add_subplot(g[i, j + 2])
                out = res[(sr, ss)]
                if residual:
                    im = show_diff(ax, crop(out, name) - crop(ref, name))
                else:
                    show(ax, dcrop(out, name))
                if i == 0:
                    ax.set_title(rf"$\sigma_s={ss}$", pad=3)
                if j == 0:
                    ax.set_ylabel(rf"$\sigma_r={sr:.2f}$", labelpad=2)
    fig.suptitle(title, y=1.0)
    save(fig, fname)


def fig_sweep(A, guide, prefix, fname, name_tex, sym, ref, ref_tex):
    """Output grid, plus a grid of differences (output - ref)."""
    res = sweep_results(A, guide, prefix)
    _sweep_figure(A, res, fname,
                  rf"{name_tex}: $\sigma_r$ (rows) $\times$ $\sigma_s$ (columns)", False)
    _sweep_figure(A, res, fname.replace(".png", "_residual.png"),
                  rf"{name_tex}: difference {sym} $-$ {ref_tex}", True, ref=ref, ref_tex=ref_tex)


def fig_detail(A, F, P):
    name = list(CROPS)[0]
    eps0 = CHOSEN["eps"]
    layer = luma((F + eps0) / (P["F_base"] + eps0))
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.55),
                             gridspec_kw=dict(wspace=0.04, hspace=0.16))
    show(axes[0, 0], crop(F, name), "Flash $F$")
    show(axes[0, 1], crop(P["F_base"], name), r"$F^{Base}$")
    show(axes[0, 2], dcrop(P["A_nr"], name), r"$A^{NR}$")
    ax = axes[0, 3]
    im = ax.imshow(crop(layer, name), cmap="magma", vmin=0.5, vmax=1.5)
    clean_ax(ax)
    ax.set_title(r"detail $(F+\epsilon)/(F^{Base}+\epsilon)$", pad=3)
    inset_colorbar(fig, ax, im)
    for j, eps in enumerate(SWEEP_EPS):
        det = P["A_nr"] * (F + eps) / (P["F_base"] + eps)
        show(axes[1, j], dcrop(det, name), rf"$A^{{Detail}},\ \epsilon={eps:g}$")
        if eps == eps0:
            frame(axes[1, j], "#d62728")
    im = show_diff(axes[1, 3], crop(P["A_detail"], name) - crop(P["A_nr"], name),
                   r"$A^{Detail} - A^{NR}$")
    inset_colorbar(fig, axes[1, 3], im)
    save(fig, "p1_detail_transfer.png")


def fig_masks(A, F, P):
    blue, red = "#1f77ff", "#ff2d2d"
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3),
                             gridspec_kw=dict(wspace=0.03, hspace=0.14))
    f_s = small(F)
    sm = lambda m: small(m.astype(np.float32)) > 0.5
    show(axes[0, 0], f_s, "Flash $F$")
    show(axes[0, 1], small(P["shadow_raw"].astype(np.float32)), "Shadow (raw)", cmap="gray")
    show(axes[0, 2], small(P["shadow"].astype(np.float32)), "Shadow (cleaned)", cmap="gray")
    ax = axes[0, 3]
    im = ax.imshow(small(P["M"]), cmap="gray", vmin=0, vmax=1)
    clean_ax(ax)
    ax.set_title("Feathered $M$", pad=3)
    for k, (name, color) in enumerate(zip(CROPS, CROP_COLORS)):
        show(axes[1, 2 * k], overlay(dcrop(F, name), [(crop(P["shadow"], name), blue),
                                                      (crop(P["spec"], name), red)]),
             name + ": $M$")
        frame(axes[1, 2 * k], color)
        show(axes[1, 2 * k + 1], dcrop(P["A_final"], name), name + r": $A^{Final}$")
        frame(axes[1, 2 * k + 1], color)
    handles = [Rectangle((0, 0), 1, 1, fc=blue, alpha=0.6),
               Rectangle((0, 0), 1, 1, fc=red, alpha=0.6)]
    cov = lambda m: 100 * (m > 0.5).mean()
    fig.legend(handles, [f"shadow ({cov(P['shadow']):.2f}% of pixels)",
                         f"specularity ({cov(P['spec']):.3f}% of pixels)"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    save(fig, "p1_masks.png")


def _stage_axes(ncols, W, extra_right=0.0):
    """A row of downsampled full frames above one row per crop, without dead space."""
    nc = len(CROPS)
    gap = 0.03
    cw = (W - extra_right) / (ncols + gap * (ncols - 1))
    hr = [2 / 3] + [1] * nc
    top_pad = 0.28
    H = cw * (sum(hr) + gap * nc) + top_pad
    fig = plt.figure(figsize=(W, H))
    g = fig.add_gridspec(1 + nc, ncols, wspace=gap, hspace=gap, height_ratios=hr,
                         left=0, right=1 - extra_right / W, bottom=0, top=1 - top_pad / H)
    axes = np.array([[fig.add_subplot(g[i, j]) for j in range(ncols)]
                     for i in range(1 + nc)])
    return fig, axes


def fig_stages(A, P):
    imgs = {"A": A, **P}
    fig, axes = _stage_axes(len(STAGES), 7.0)
    for j, (k, t) in enumerate(STAGES):
        show(axes[0, j], small(imgs[k], 0.15), t)
        draw_crop_boxes(axes[0, j], 0.15)
        for i, (name, color) in enumerate(zip(CROPS, CROP_COLORS)):
            show(axes[i + 1, j], dcrop(imgs[k], name))
            frame(axes[i + 1, j], color)
    save(fig, "p1_stages.png")


def fig_stage_diffs(A, P, pairs=None):
    imgs = {"A": A, **P}
    if pairs is None:
        pairs = [("A_base", "A"), ("A_nr", "A"), ("A_detail", "A_nr"), ("A_final", "A_detail")]
    lab = dict(STAGES)
    lab["A"] = "$A$"
    right = 0.8
    fig, axes = _stage_axes(len(pairs), 7.0, extra_right=right)
    for j, (a, b) in enumerate(pairs):
        d = imgs[a] - imgs[b]
        im = show_diff(axes[0, j], small(d, 0.15), f"{lab[a]} $-$ {lab[b]}")
        for i, (name, color) in enumerate(zip(CROPS, CROP_COLORS)):
            show_diff(axes[i + 1, j], crop(d, name))
            frame(axes[i + 1, j], color)
    cax = fig.add_axes([1 - right / 7.0 + 0.02, 0.08, 0.018, 0.75])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(DIFF_LABEL, fontsize=8)
    save(fig, "p1_stage_differences.png")


def stats(A, P):
    out = {}
    for k, _ in STAGES[1:]:
        d = P[k] - A
        out[k] = dict(mean_abs_vs_A=float(np.abs(d).mean()),
                      rms_vs_A=float(np.sqrt((d ** 2).mean())))
    out["mask_coverage_percent"] = {k: float(100 * (P[k] > 0.5).mean())
                                    for k in ["shadow_raw", "shadow", "spec_raw", "spec", "M"]}
    out["chosen_parameters"] = CHOSEN
    path = os.path.join(CACHE_DIR, "p1_stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved", path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    A = readimg("data/lamp/lamp_ambient.tif")
    F = readimg("data/lamp/lamp_flash.tif")

    P = run_pipeline(A, F)
    fig_inputs(A, F)
    fig_sweep(A, A, "Abase", "p1_sweep_base.png", r"Bilateral $A^{Base}$",
              r"$A^{Base}$", A, "$A$")
    fig_sweep(A, F, "Anr", "p1_sweep_nr.png", r"Joint bilateral $A^{NR}$",
              r"$A^{NR}$", P["A_base"], r"$A^{Base}$")
    fig_detail(A, F, P)
    fig_masks(A, F, P)
    fig_stages(A, P)
    fig_stage_diffs(A, P)
    stats(A, P)


if __name__ == "__main__":
    main()
