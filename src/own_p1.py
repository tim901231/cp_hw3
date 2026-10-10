"""Problem 3, pair 1: run the Problem 1 pipeline on my own flash/no-flash pair.

Run from the repository root:
    python src/own_p1.py

Uses the same parameters as main.p1(). The pair was taken on a tripod and is used
without alignment.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import report_figures_p1 as R
from main import readimg, save_jpg, bilateral_filtering, brute_force_bilateral

AMBIENT = "data/own/desk_ambient.jpg"
FLASH = "data/own/desk_flash.jpg"
OUT_DIR = "output/p3_pair1"          # figures
RESULT_DIR = "results/problem3_pair1"

R.FIG_DIR = OUT_DIR
# Zoom regions in 6000-pixel-wide coordinates (scaled to the actual image width)
CROPS_6000 = {"Scarf": (3150, 1750, 600, 600), "Beanie": (2450, 2750, 600, 600)}
R.CROP_GAIN = {"Scarf": 3.0, "Beanie": 3.0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    A = readimg(AMBIENT)
    F = readimg(FLASH)
    k = A.shape[1] / 6000
    R.CROPS = {n: tuple(round(v * k) for v in box) for n, box in CROPS_6000.items()}

    A_base = bilateral_filtering(A, A, sigma_s=16, sigma_r=0.05)
    F_base = bilateral_filtering(F, F, sigma_s=16, sigma_r=0.05)
    A_nr = brute_force_bilateral(A, F, sigma_s=4, sigma_r=0.001)
    A_detail = A_nr * (F + 0.02) / (F_base + 0.02)
    masks = R.compute_mask(A, F)
    M = masks["M"][..., None]
    A_final = (1 - M) * A_detail + M * A_base

    P = dict(A_base=A_base, F_base=F_base, A_nr=A_nr, A_detail=A_detail, A_final=A_final, **masks)
    for k in ["A_base", "A_nr", "A_detail", "A_final"]:
        save_jpg(os.path.join(RESULT_DIR, f"{k}.jpg"), P[k])

    R.fig_inputs(A, F)
    R.fig_stages(A, P)
    R.fig_stage_diffs(A, P, pairs=[("A_base", "A"), ("A_nr", "A"),
                                   ("A_nr", "A_base"), ("A_detail", "A_nr")])
    R.fig_masks(A, F, P)
    R.fig_detail(A, F, P)
    print("mask coverage (%):", {k: round(100 * float((masks[k] > 0.5).mean()), 2)
                                 for k in ["shadow", "spec", "M"]})


if __name__ == "__main__":
    main()
