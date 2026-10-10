# CP HW3: Flash/No-Flash Photography

15-663 Computational Photography, Homework 3. Bilateral-filtering denoising
(Petschnigg et al.), gradient-domain fusion (Agrawal et al.), my own
flash/no-flash pairs, and two bonus problems (reflection removal and
gradient-domain non-photorealistic rendering).

The report is [`report/report.pdf`](report/report.pdf) (LaTeX source in `report/report.tex`).

## Setup

Tested with Python 3.8 and:

```
numpy 1.24  scipy 1.10  opencv-python 5.0  Pillow 10.4  matplotlib 3.7
```

```bash
pip install numpy scipy opencv-python pillow matplotlib
```

The lamp and museum pairs from the homework ZIP are not committed. Copy them to:

```
data/lamp/lamp_ambient.tif   data/lamp/lamp_flash.tif
data/museum/museum_ambient.png   data/museum/museum_flash.png
```

My own photographs are committed in `data/own/` (see [My captures](#my-captures)).

## Usage

Run everything from the repository root:

```bash
python src/main.py <command>
```

| Command | What it does | Final images |
|---|---|---|
| `p1` | Problem 1: basic and joint bilateral filtering, detail transfer, shadow/specularity masking on the lamp pair | `results/problem1/` |
| `p2` | Problem 2: re-integration test and gradient-domain fusion on the museum pair | `results/problem2/` |
| `p3-pair1` | Problem 3, pair 1: Problem 1 pipeline on my desk pair | `results/problem3_pair1/` |
| `p3-pair2` | Problem 3, pair 2: gradient-domain fusion on my acrylic-card pair | `results/problem3_pair2/` |
| `bonus-reflection` | Bonus 4: reflection removal (Agrawal et al., Observation Deck) on my window pair | `results/bonus_reflection/` |
| `bonus-npr` | Bonus 5: gradient-domain NPR on three photos, plus a parameter sweep | `results/bonus_npr/` |
| `figures-p1` | Report figures for Problem 1 (parameter sweeps, masks, comparisons) | `report/figures/` |
| `figures-p2` | Report figures for Problem 2 (re-integration, gradients, boundary/initialization tests) | `report/figures/` |
| `all` | All of the above | |

`python src/main.py -h` lists the commands. Intermediate figures and caches go
to `output/` (not committed); delete `output/cache/` to force recomputation.

Approximate run times on a laptop: `p1` 1 min, `p2` 1 min, `p3-pair1` < 1 min,
`p3-pair2` 3 min, `bonus-reflection` 3 min, `bonus-npr` 15 min, `figures-p1`
3 min (first run), `figures-p2` 8 min (first run).

## Code

| File | Contents |
|---|---|
| `src/main.py` | Core implementation and command-line entry point: `bilateral_filtering` (PiecewiseBilateral, Durand and Dorsey), `brute_force_bilateral`, `shadow_specular_mask`, `clean_mask`, the gradient operators `grad` / `divergence` / `laplacian`, the CGD Poisson solver `gradient_cgd` (Algorithm 1), `fused_gradient` and `fuse` (Agrawal et al. Eqs. 5, 11, 12), `project` and `reflection_free_gradient` (Eqs. 9, 13, 14), and `p1()` / `p2()` |
| `src/own_p1.py` | Problem 3, pair 1 |
| `src/own_p2.py` | Problem 3, pair 2 |
| `src/own_bonus.py` | Bonus 4: reflection removal |
| `src/bonus_npr.py` | Bonus 5: NPR with a Canny-based long-edge detector |
| `src/report_figures_p1.py` | Figures and parameter sweeps for Problem 1 |
| `src/report_figures_p2.py` | Figures and experiments for Problem 2 |

## Selected parameters

| Problem | Parameters |
|---|---|
| 1, basic bilateral ($A^{Base}$) | piecewise, $\sigma_s=16$, $\sigma_r=0.05$, $\lambda=0.001$ |
| 1, joint bilateral ($A^{NR}$) | brute force, $\sigma_s=4$, $\sigma_r=0.001$ (0.1% of the range, as in Petschnigg et al.) |
| 1, detail transfer | $F^{Base}$: piecewise, $\sigma_s=16$, $\sigma_r=0.05$; $\epsilon=0.02$ |
| 1, masking | linear luminance, ambient ISO-normalized (1600 vs. 200); $\tau_{shad}=0$; specularity $Y_F>0.95$; per-mask opening $3\times3$, closing $5\times5$, dilation $3\times3$; union feathered with Gaussian $\sigma=2$ |
| 2, fusion | $\sigma=40$, $\tau_s=0.9$; CGD $\epsilon=10^{-3}$, $N=3000$; initialization: average of ambient and flash; boundary: ambient |
| 3, pair 2 | as Problem 2 but $\tau_s=0.7$; images resized to 1500 px wide |
| Bonus 4 | $\sigma=40$, $\tau_{ue}=0.1$; boundary and initialization from $H=a+\Phi'$; 1500 px wide |
| Bonus 5 | bilateral pre-smoothing ($\sigma_s=3$, $\sigma_r=0.1$); Canny thresholds $0.66m$, $1.33m$ ($m$: median); edge chains $\geq80$ px; $\alpha=2$, $\beta=0.3$; 1000 px wide |

## My captures

All camera pairs were taken with a Nikon D3500 on a tripod at $f/4$; the two
images of a pair are used without alignment. The committed files in
`data/own/` are downsampled copies (2000 px wide) of the original
6000 x 4000 captures, which are kept locally because of their size (EXIF data
is not preserved, so the settings are listed here).

| Files | Use | Ambient | Flash |
|---|---|---|---|
| `desk_ambient.jpg`, `desk_flash.jpg` | Problem 3, pair 1 | ISO 1600, 1/4 s | ISO 200, 1/4 s |
| `cards_ambient.jpg`, `cards_flash.jpg` | Problem 3, pair 2 | ISO 200, 4 s | ISO 200, 1/4 s |
| `window_ambient.jpg`, `window_flash.jpg` | Bonus 4 | ISO 200, 15 s | ISO 200, 1/4 s |
| `npr_1.jpg` ... `npr_3.jpg` | Bonus 5 | phone photos, 1500 px wide (rotated upright) | |

**Note:** the Problem 3 pair 1 figures in the report were computed at full
resolution. Running `p3-pair1` on the committed 2000-px copies gives
lower-resolution results; since the filter widths are in pixels, they look
slightly smoother. All other experiments downsample the images anyway (to
1500 or 1000 px), so they are essentially unaffected.

## Results

| Folder | Files |
|---|---|
| `results/problem1/` | `A_base.jpg`, `A_NR.jpg`, `F_base.jpg`, `A_detail.jpg`, `A_final.jpg`, `mask.jpg` |
| `results/problem2/` | `reintegrated.jpg`, `museum_fused.jpg` |
| `results/problem3_pair1/` | `A_base.jpg`, `A_nr.jpg`, `A_detail.jpg`, `A_final.jpg` |
| `results/problem3_pair2/` | `cards_fused.jpg` |
| `results/bonus_reflection/` | `window_H.jpg` (ambient + flash), `window_reflection_removed.jpg`, `window_reflection_layer.jpg` |
| `results/bonus_npr/` | `photo1_npr.jpg`, `photo2_npr.jpg`, `photo3_npr.jpg` |

## Acknowledgement

The visualizations and the writing of the report were prepared in
collaboration with Claude Code (Claude Opus 5.5).
