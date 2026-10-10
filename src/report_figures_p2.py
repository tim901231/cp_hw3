"""Generate Problem 2 report figures (museum flash/no-flash pair).

Run from the repository root:
    python src/report_figures_p2.py

Fused images are cached in output/cache/ (delete to recompute).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from main import readimg, grad, divergence, laplacian, gradient_cgd, fused_gradient

FIG_DIR = "report/figures"
CACHE_DIR = "output/cache"

SIGMA, TAU_S = 40, 0.9            # Agrawal et al. recommended values
EPS, N_ITER = 1e-3, 3000          # CGD convergence parameters for fusion
SWEEP_SIGMA = [10, 40, 100]
SWEEP_TAU = [0.7, 0.8, 0.9]
INITS = ["zero", "ambient", "flash", "average"]
BOUNDARIES = ["ambient", "flash", "average"]

# Zoom regions (x, y, w, h)
CROPS = {"Flash highlight": (140, 270, 250, 190), "Face": (560, 260, 220, 165)}
CROP_COLORS = ["#ffcc33", "#33ccff"]
GRAD_GAIN = 0.15                  # gradient display: 0.5 + g / (2 * GRAD_GAIN)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


# ---------------------------------------------------------------- helpers ----
def crop(img, key):
    x, y, w, h = CROPS[key]
    return img[y:y + h, x:x + w]


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


def show_signed(ax, d, lim, title=None):
    ax.imshow(np.clip(0.5 + d / (2 * lim), 0, 1), interpolation="nearest")
    clean_ax(ax)
    if title:
        ax.set_title(title, pad=3)


def draw_crop_boxes(ax):
    for (x, y, w, h), c in zip(CROPS.values(), CROP_COLORS):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=c, lw=1.4))


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def per_channel(fn, img):
    return np.stack([fn(img[..., c]) for c in range(3)], axis=-1)


def run_fusion(a, f, init, boundary, sigma=SIGMA, tau_s=TAU_S, eps=EPS, N=N_ITER):
    """Fused RGB image and the per-channel CGD residual histories (cached)."""
    path = os.path.join(CACHE_DIR, f"p2_{init}_{boundary}_s{sigma}_t{tau_s}_e{eps}_N{N}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["img"], list(d["hist"])
    images = {"ambient": a, "flash": f, "average": 0.5 * (a + f), "zero": np.zeros_like(a)}
    out = np.zeros_like(a)
    hist = []
    for c in range(3):
        g, _ = fused_gradient(a[..., c], f[..., c], sigma, tau_s)
        out[..., c], h = gradient_cgd(divergence(g), images[init][..., c],
                                      images[boundary][..., c], eps, N)
        hist.append(h)
    np.savez(path, img=out, hist=np.array(hist, dtype=object))
    print(f"  fused init={init} boundary={boundary} sigma={sigma} tau={tau_s}: "
          f"{[len(h) - 1 for h in hist]} iterations")
    return out, hist


# ---------------------------------------------------------------- figures ----
def fig_reintegration(a):
    """Differentiate and re-integrate the ambient image (all channels)."""
    I = a
    D = per_channel(laplacian, I)
    res = {}
    for eps, N in [(1e-3, 3000), (1e-6, 5000)]:
        out = np.zeros_like(I)
        hist = []
        for c in range(3):
            out[..., c], h = gradient_cgd(D[..., c], np.zeros_like(I[..., c]), I[..., c], eps, N)
            hist.append(h)
        res[eps] = (out, hist)

    fig = plt.figure(figsize=(7.0, 3.9))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.05, hspace=0.22)
    show(fig.add_subplot(g[0, 0]), I, "Original $I$")
    out, hist = res[1e-3]
    show(fig.add_subplot(g[0, 1]), out, r"$I^*$ ($\epsilon=10^{-3}$)")
    ax = fig.add_subplot(g[0, 2])
    err = np.abs(out - I).max(axis=-1)
    im = ax.imshow(err, cmap="magma", vmin=0, vmax=0.01)
    clean_ax(ax)
    ax.set_title(r"$|I^*-I|$ ($\epsilon=10^{-3}$)", pad=3)
    cax = ax.inset_axes([1.03, 0, 0.05, 1])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.ax.tick_params(labelsize=7, length=2)

    ax = fig.add_subplot(g[1, :])
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for eps, ls in [(1e-6, "-")]:
        for c, col in enumerate(colors):
            ax.semilogy(res[eps][1][c], ls, color=col, lw=1.2, label="RGB"[c] + " channel")
    for eps in [1e-3, 1e-6]:
        ax.axhline(eps, color="0.5", lw=0.8, ls=":")
        ax.text(10, eps * 1.4, rf"$\epsilon=10^{{{int(np.log10(eps))}}}$", fontsize=7.5, color="0.4")
    ax.set_xlabel("CGD iteration $n$")
    ax.set_ylabel(r"$\sqrt{\langle r,r\rangle}$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "p2_reintegration.png")
    return {eps: (float(np.abs(o - I).max()), [len(h) - 1 for h in hs]) for eps, (o, hs) in res.items()}


def fig_inputs(a, f):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.03))
    for ax, img, t in zip(axes, [a, f], [r"Ambient $a$", r"Flash $\Phi'$"]):
        show(ax, img, t)
        draw_crop_boxes(ax)
    handles = [Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=1.4) for c in CROP_COLORS]
    fig.legend(handles, list(CROPS), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    save(fig, "p2_inputs.png")


def fig_gradients(a, f):
    """x and y components of the ambient, flash and fused gradient fields."""
    fields = {"a": [], "f": [], "fused": []}
    Ms, ws = [], []
    for c in range(3):
        g, q = fused_gradient(a[..., c], f[..., c], SIGMA, TAU_S)
        fields["a"].append(q["g_a"])
        fields["f"].append(q["g_f"])
        fields["fused"].append(g)
        Ms.append(q["M"])
        ws.append(q["w_s"])
    fields = {k: np.stack(v, axis=-1) for k, v in fields.items()}   # (2, H, W, 3)
    names = {"a": r"$\nabla a$", "f": r"$\nabla\Phi'$", "fused": r"$\nabla\Phi^*$"}

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.35), gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for j, k in enumerate(["a", "f", "fused"]):
        show_signed(axes[0, j], fields[k][1], GRAD_GAIN, names[k] + r": $x$ component")
        show_signed(axes[1, j], fields[k][0], GRAD_GAIN, names[k] + r": $y$ component")
    save(fig, "p2_gradients.png")

    # Coherency map and saturation weight (mean over channels)
    M = np.mean(Ms, axis=0)
    w = np.mean(ws, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.18))
    for ax, img, t in zip(axes, [M, w], ["Coherency map $M$", rf"Saturation weight $w_s$ ($\sigma={SIGMA}$, $\tau_s={TAU_S}$)"]):
        im = ax.imshow(img, cmap="magma", vmin=0, vmax=1)
        clean_ax(ax)
        ax.set_title(t, pad=3)
        cax = ax.inset_axes([1.02, 0, 0.035, 1])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7, length=2)
    save(fig, "p2_weights.png")


def fig_sweep(a, f):
    """Fused result (highlight crop) for a grid of sigma and tau_s, with w_s below."""
    name = "Flash highlight"
    ns, nt = len(SWEEP_SIGMA), len(SWEEP_TAU)
    fig, axes = plt.subplots(nt, ns, figsize=(5.2, 5.2 * 190 / 250 * nt / ns + 0.35),
                             gridspec_kw=dict(wspace=0.03, hspace=0.03))
    for i, tau in enumerate(SWEEP_TAU):
        for j, sig in enumerate(SWEEP_SIGMA):
            out, _ = run_fusion(a, f, "average", "ambient", sigma=sig, tau_s=tau)
            ax = axes[i, j]
            show(ax, crop(out, name))
            if sig == SIGMA and tau == TAU_S:
                frame(ax, "#d62728")
            if i == 0:
                ax.set_title(rf"$\sigma={sig}$", pad=3)
            if j == 0:
                ax.set_ylabel(rf"$\tau_s={tau}$", labelpad=2)
    save(fig, "p2_sweep.png")


def fig_bc_init(a, f):
    """Boundary conditions change the result; initializations only change convergence."""
    results = {(i, b): run_fusion(a, f, i, b) for i in INITS for b in BOUNDARIES}
    ref = results[("zero", "ambient")][0]
    init_diff = max(np.abs(results[(i, b)][0] - results[("zero", b)][0]).max()
                    for i in INITS for b in BOUNDARIES)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 7.0 / 3 * 638 / 905 * 2 + 0.35),
                             gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for c, bnd in enumerate(BOUNDARIES):
        out = results[("average", bnd)][0]
        show(axes[0, c], out, f"boundary: {bnd}")
        if bnd == "ambient":
            show(axes[1, c], np.full_like(out, 0.5))
            axes[1, c].text(0.5, 0.5, "reference", ha="center", va="center",
                            transform=axes[1, c].transAxes, color="0.3")
        else:
            show_signed(axes[1, c], out - ref, 0.15, f"{bnd} $-$ ambient")
    save(fig, "p2_bc_init.png")

    # Convergence for each initialization (boundary = ambient, green channel)
    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    for init, col in zip(INITS, ["0.2", "#d62728", "#1f77b4", "#2ca02c"]):
        h = results[(init, "ambient")][1][1]
        ax.semilogy(h, color=col, lw=1.2, label=f"init: {init} ({len(h) - 1} it.)")
    ax.axhline(EPS, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("CGD iteration $n$")
    ax.set_ylabel(r"$\sqrt{\langle r,r\rangle}$")
    ax.legend(frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "p2_convergence.png")
    print(f"max difference between initializations (same boundary): {init_diff:.4f}")
    for b in BOUNDARIES:
        print(f"  boundary={b}: iterations (max over RGB) per init:",
              {i: max(len(h) - 1 for h in results[(i, b)][1]) for i in INITS})
    return results


def fig_final(a, f, out):
    fig = plt.figure(figsize=(7.0, 2.95))
    g = fig.add_gridspec(2, 3, height_ratios=[1.65, 0.88], wspace=0.03, hspace=0.08)
    for j, (img, t) in enumerate([(a, r"Ambient $a$"), (f, r"Flash $\Phi'$"), (out, r"Fused $\Phi^*$")]):
        ax = fig.add_subplot(g[0, j])
        show(ax, img, t)
        draw_crop_boxes(ax)
    sub = g[1, :].subgridspec(1, 6, wspace=0.03)
    k = 0
    for name, color in zip(CROPS, CROP_COLORS):
        for img in (a, f, out):
            ax = fig.add_subplot(sub[0, k])
            show(ax, crop(img, name))
            frame(ax, color)
            k += 1
    save(fig, "p2_final.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    a = readimg("data/museum/museum_ambient.png").astype(np.float64)
    f = readimg("data/museum/museum_flash.png").astype(np.float64)

    print("re-integration:", fig_reintegration(a))
    fig_inputs(a, f)
    fig_gradients(a, f)
    results = fig_bc_init(a, f)
    fig_sweep(a, f)
    fig_final(a, f, results[("average", "ambient")][0])


if __name__ == "__main__":
    main()
