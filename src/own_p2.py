"""Problem 3, pair 2: gradient-domain fusion of my own flash/no-flash pair.

Run from the repository root:
    python src/own_p2.py

Uses the same parameters as Problem 2. Images are downsampled to WIDTH pixels
(CGD on the full 24-megapixel images would be very slow). The pair was taken on
a tripod and is used without alignment.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import report_figures_p2 as R
from main import readimg, save_jpg

AMBIENT = "data/own/cards_ambient.jpg"
FLASH = "data/own/cards_flash.jpg"
OUT_DIR = "output/p3_pair2"          # figures
RESULT_DIR = "results/problem3_pair2"
WIDTH = 1500

R.FIG_DIR = OUT_DIR
R.CACHE_DIR = "output/cache/p3_pair2"
R.CROPS = {"Acrylic glare": (510, 230, 330, 220), "Mug": (0, 130, 330, 220)}
R.N_ITER = 5000
R.TAU_S = 0.7                     # lower than the paper's 0.9 to cover the unsaturated halo
TAU_COMPARE = [0.9, 0.7]


def load(path):
    img = readimg(path)
    h = round(img.shape[0] * WIDTH / img.shape[1])
    return cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA).astype(np.float64)


def fig_tau(a, f):
    """Crops of the fused result for the paper's tau_s and the selected one."""
    import matplotlib.pyplot as plt
    outs = {t: R.run_fusion(a, f, "average", "ambient", tau_s=t)[0] for t in TAU_COMPARE}
    cols = [("Ambient $a$", a), (r"Flash $\Phi'$", f)] + \
           [(rf"$\Phi^*$, $\tau_s={t}$", outs[t]) for t in TAU_COMPARE]
    fig, axes = plt.subplots(len(R.CROPS), len(cols), figsize=(7.0, 7.0 / len(cols) * 220 / 330 * len(R.CROPS) + 0.3),
                             gridspec_kw=dict(wspace=0.03, hspace=0.04))
    for i, (name, color) in enumerate(zip(R.CROPS, R.CROP_COLORS)):
        for j, (t, img) in enumerate(cols):
            ax = axes[i, j]
            R.show(ax, R.crop(img, name), t if i == 0 else None)
            R.frame(ax, color)
    R.save(fig, "p2_tau_compare.png")


def main():
    os.makedirs(R.CACHE_DIR, exist_ok=True)
    a = load(AMBIENT)
    f = load(FLASH)

    R.fig_inputs(a, f)
    R.fig_gradients(a, f)
    fused, hist = R.run_fusion(a, f, "average", "ambient", tau_s=R.TAU_S)
    print("iterations per channel:", [len(h) - 1 for h in hist])
    R.fig_final(a, f, fused)
    fig_tau(a, f)
    save_jpg(os.path.join(RESULT_DIR, "cards_fused.jpg"), fused)


if __name__ == "__main__":
    main()
