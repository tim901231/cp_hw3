"""Computational Photography HW3: flash/no-flash photography.

Core implementation (bilateral filtering, gradient-domain processing) plus a
command-line entry point for every problem. Run from the repository root:

    python src/main.py p1                 # Problem 1: bilateral filtering (lamp)
    python src/main.py p2                 # Problem 2: gradient-domain fusion (museum)
    python src/main.py p3-pair1           # Problem 3, pair 1: my bilateral-filtering pair
    python src/main.py p3-pair2           # Problem 3, pair 2: my gradient-fusion pair
    python src/main.py bonus-reflection   # Bonus 4: reflection removal
    python src/main.py bonus-npr          # Bonus 5: gradient-domain NPR
    python src/main.py figures-p1         # report figures for Problem 1
    python src/main.py figures-p2         # report figures for Problem 2
    python src/main.py all                # everything above

Final images are written to results/, intermediate figures and caches to output/.
"""
import argparse
import math
import os
import sys

import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import interpn
from scipy.signal import convolve2d

def bilateral_filtering(img1, img2, sigma_s=4, sigma_r=0.05, lambda_0 = 0.001):
    H, W, _ = img1.shape
    out = np.zeros_like(img1)
    # sigma_s = sigma_s * (H + W)
    minI = img2.min() - lambda_0
    maxI = img2.max() + lambda_0
    NB_SEG = math.ceil((maxI-minI) / sigma_r)
    J = []
    points = []

    for j in range(NB_SEG+1):
        ij = minI + j * (maxI - minI) / NB_SEG
        Gj = np.exp(-1 * (img2 - ij)**2 / (2*sigma_r**2))
        Kj = cv2.GaussianBlur(Gj, (0, 0), sigma_s)
        Hj = Gj * img1
        H_j = cv2.GaussianBlur(Hj, (0, 0), sigma_s)
        Jj = H_j / (Kj + 1e-8)

        J.append(Jj)
        points.append(ij)

    J = np.stack(J)
    points = (np.stack(points), np.arange(H), np.arange(W))

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    for c in range(3):
        xi = np.stack([img2[:, :, c], yy, xx], axis=-1)
        out[:, :, c] = interpn(points, J[:, :, :, c], xi)

    return out

def brute_force_bilateral(img1, img2, sigma_s=5, sigma_r=0.001, truncate=3.0):
    """Direct (non-piecewise) bilateral filter of img1 with range weights from img2.
    Cost grows with the window size (sigma_s) but not with 1/sigma_r, so it suits
    the small sigma_r used for joint bilateral filtering. Channels are filtered
    separately, as in bilateral_filtering."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    H, W, _ = img1.shape
    r = int(math.ceil(truncate * sigma_s))
    pad = ((r, r), (r, r), (0, 0))
    src = np.pad(img1, pad, mode="reflect")
    guide = np.pad(img2, pad, mode="reflect")
    inv_2sr2 = np.float32(-1.0 / (2 * sigma_r ** 2))

    num = np.zeros_like(img1)
    den = np.zeros_like(img1)
    w = np.empty_like(img1)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if d2 > r * r:  # circular window
                continue
            ws = np.float32(math.exp(-d2 / (2 * sigma_s ** 2)))
            g = guide[r + dy:r + dy + H, r + dx:r + dx + W]
            np.subtract(g, img2, out=w)
            np.square(w, out=w)
            w *= inv_2sr2
            np.exp(w, out=w)
            w *= ws
            den += w
            w *= src[r + dy:r + dy + H, r + dx:r + dx + W]
            num += w
    return num / den

def readimg(filepath):
    srgb = np.asarray(Image.open(filepath).convert("RGB"), dtype=np.float32) / 255.0
    # linear = np.where(
    #     srgb <= 0.0404482,
    #     srgb / 12.92,
    #     ((srgb + 0.055) / 1.055) ** 2.4
    # )
    return srgb

def save_jpg(filename, data, scale=1.0):
    data = np.asarray(data, dtype=np.float32) * scale

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    img_display = np.clip(data, 0, 1)
    # img_display = np.where(
    #     img_display <= 0.0031308,
    #     12.92 * img_display,
    #     1.055 * img_display ** (1 / 2.4) - 0.055,
    # )
    img_display = np.round(img_display * 255).astype(np.uint8)

    Image.fromarray(img_display).save(
        filename, format="JPEG", quality=95, subsampling=0
    )

def to_linear(rgb):
    return np.where(
        rgb <= 0.0404482,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )

def linear_rgb_to_luminance(rgb):
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )

def clean_mask(M, open_k=3, close_k=5, dilate_k=3):
    """Petschnigg et al. Sec. 4.3: erode speckles (opening), fill holes (closing),
    then dilate for a conservative estimate."""
    M = M.astype(np.uint8)
    M = cv2.morphologyEx(M, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    M = cv2.dilate(M, np.ones((dilate_k, dilate_k), np.uint8))
    return M.astype(bool)


def shadow_specular_mask(ambient_img, flash_img, iso_ambient=1600, iso_flash=200,
                         tau_shadow=0.00125, spec_thresh=0.95, feather_sigma=2):
    """Soft mask M in [0, 1] marking flash shadows and specularities."""
    Y_flash = linear_rgb_to_luminance(to_linear(flash_img))
    Y_ambient = linear_rgb_to_luminance(to_linear(ambient_img))

    # ISO-normalize the ambient image to the flash image (equal exposure times)
    M_shadow = Y_flash - Y_ambient * iso_flash / iso_ambient <= tau_shadow
    # 95% of the sensor output range (images are normalized to [0, 1])
    M_spec = Y_flash > spec_thresh

    M = clean_mask(M_shadow) | clean_mask(M_spec)

    # Feather the edges to avoid visible seams
    if feather_sigma > 0:
        return cv2.GaussianBlur(M.astype(np.float32), (0, 0), feather_sigma)
    return M.astype(np.float32)


def p1(out_dir="results/problem1"):
    """Problem 1 on the lamp pair with the selected parameters (see the report)."""
    ambient_img = readimg("data/lamp/lamp_ambient.tif")
    flash_img = readimg("data/lamp/lamp_flash.tif")

    # Basic bilateral filtering (piecewise implementation)
    A_base = bilateral_filtering(ambient_img, ambient_img, sigma_s=16, sigma_r=0.05)
    # Joint bilateral filtering; brute force because sigma_r = 0.001 would need
    # about 1000 segments in the piecewise implementation
    A_nr = brute_force_bilateral(ambient_img, flash_img, sigma_s=4, sigma_r=0.001)

    # Detail transfer
    F_base = bilateral_filtering(flash_img, flash_img, sigma_s=16, sigma_r=0.05)
    epsilon = 0.02
    A_detail = A_nr * (flash_img + epsilon) / (F_base + epsilon)

    # Shadow and specularity masking
    M = shadow_specular_mask(ambient_img, flash_img, tau_shadow=0.0)[..., np.newaxis]
    A_final = (1 - M) * A_detail + M * A_base

    for name, img in [("A_base", A_base), ("A_NR", A_nr), ("F_base", F_base),
                      ("A_detail", A_detail), ("A_final", A_final), ("mask", np.repeat(M, 3, axis=2))]:
        save_jpg(os.path.join(out_dir, f"{name}.jpg"), img)
    print(f"Problem 1 results saved to {out_dir}/ (mask covers {100 * (M > 0.5).mean():.2f}% of pixels)")


LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def laplacian(img):
    return convolve2d(img, LAPLACIAN_KERNEL, mode='same', boundary='fill', fillvalue=0)


def grad(img):
    """Backward differences with zero padding: returns [I_y, I_x] (axis 0, axis 1)."""
    return np.stack([np.diff(img, axis=0, prepend=0), np.diff(img, axis=1, prepend=0)])


def divergence(g):
    """Forward differences of a [g_y, g_x] field, so that divergence(grad(I)) equals
    laplacian(I) everywhere except on the image boundary."""
    return np.diff(g[0], axis=0, append=0) + np.diff(g[1], axis=1, append=0)


def gradient_cgd(D, I_init, I_boundary, epsilon=1e-3, N=1000):
    """Algorithm 1: integrate a target divergence D with conjugate gradient descent.
    Returns the integrated image and the residual norm at every iteration."""
    B = np.zeros_like(D)
    B[1:-1, 1:-1] = 1

    I = B * I_init + (1 - B) * I_boundary
    r = B * (D - laplacian(I))
    d = r
    delta_new = np.sum(r * r)
    n = 0
    history = [np.sqrt(delta_new)]

    while np.sqrt(delta_new) > epsilon and n < N:
        q = laplacian(d)
        eta = delta_new / np.sum(d * q)
        I = I + B * (eta * d)
        r = B * (r - eta * q)
        delta_old = delta_new
        delta_new = np.sum(r * r)
        beta = delta_new / delta_old
        d = r + beta * d
        n = n + 1
        history.append(np.sqrt(delta_new))
    return I, np.array(history)


def fused_gradient(ambient, flash, sigma=40, tau_s=0.9):
    """Agrawal et al. Eqs. (5), (11), (12) for one channel. Returns the fused field
    and the intermediate quantities."""
    g_a = grad(ambient)
    g_f = grad(flash)

    # Gradient orientation coherency map
    M = np.abs(np.sum(g_a * g_f, axis=0)) / (
        np.linalg.norm(g_a, axis=0) * np.linalg.norm(g_f, axis=0) + 1e-8)

    # Saturation weight from the flash image, normalized to [0, 1]
    w_s = np.tanh(sigma * (flash - tau_s))
    w_s = (w_s - w_s.min()) / (w_s.max() - w_s.min())

    g = w_s * g_a + (1 - w_s) * (M * g_f + (1 - M) * g_a)
    return g, dict(g_a=g_a, g_f=g_f, M=M, w_s=w_s)


def project(g, onto):
    """Agrawal et al. Eq. (9): projection of gradient field g onto the direction of `onto`."""
    dot = np.sum(g * onto, axis=0)
    return onto * dot / (np.sum(onto * onto, axis=0) + 1e-8)


def reflection_free_gradient(ambient, flash, ambient_lum, sigma=40, tau_ue=0.1):
    """Agrawal et al. Sec. 4.1, Observation Deck (Eqs. 13, 14) for one channel.
    The flash causes reflections of room objects on the glass; H = ambient + flash
    has a well-lit foreground but keeps the reflections, which are removed by
    projecting grad H onto grad ambient wherever the ambient image is well exposed."""
    H = ambient + flash
    g_H = grad(H)
    g_a = grad(ambient)

    # Underexposed map from the ambient luminance, normalized to [0, 1]
    w_ue = 1 - np.tanh(sigma * (ambient_lum - tau_ue))
    w_ue = (w_ue - w_ue.min()) / (w_ue.max() - w_ue.min())

    g_proj = project(g_H, g_a)
    g = w_ue * g_H + (1 - w_ue) * g_proj
    g_reflection = (1 - w_ue) * (g_H - g_proj)      # removed component (reflection layer)
    return g, dict(H=H, w_ue=w_ue, g_H=g_H, g_a=g_a, g_reflection=g_reflection)


def fuse(ambient_img, flash_img, init="zero", boundary="ambient",
         sigma=40, tau_s=0.9, epsilon=1e-3, N=1000):
    """Fuse each RGB channel by integrating the fused gradient field.
    init: 'zero' | 'ambient' | 'flash' | 'average'; boundary: 'ambient' | 'flash' | 'average'."""
    images = {"ambient": ambient_img, "flash": flash_img,
              "average": 0.5 * (ambient_img + flash_img),
              "zero": np.zeros_like(ambient_img)}
    out = np.zeros_like(ambient_img, dtype=np.float64)
    for c in range(3):
        g, _ = fused_gradient(ambient_img[..., c], flash_img[..., c], sigma, tau_s)
        out[..., c], _ = gradient_cgd(divergence(g), images[init][..., c],
                                      images[boundary][..., c], epsilon, N)
    return out


def p2(out_dir="results/problem2"):
    """Problem 2 on the museum pair: re-integration test and gradient-domain fusion."""
    ambient_img = readimg("data/museum/museum_ambient.png").astype(np.float64)
    flash_img = readimg("data/museum/museum_flash.png").astype(np.float64)

    # Differentiate and re-integrate every channel of the ambient image
    I_star = np.zeros_like(ambient_img)
    for c in range(3):
        I = ambient_img[..., c]
        I_star[..., c], hist = gradient_cgd(laplacian(I), np.zeros_like(I), I, epsilon=1e-3, N=3000)
        print(f"  re-integration channel {c}: {len(hist) - 1} iterations")
    print("re-integration max error:", np.abs(I_star - ambient_img).max())
    save_jpg(os.path.join(out_dir, "reintegrated.jpg"), I_star)

    # Fusion with the selected parameters
    fused = fuse(ambient_img, flash_img, init="average", boundary="ambient",
                 sigma=40, tau_s=0.9, epsilon=1e-3, N=3000)
    save_jpg(os.path.join(out_dir, "museum_fused.jpg"), fused)
    print(f"Problem 2 results saved to {out_dir}/")


COMMANDS = {
    "p1": ("Problem 1: bilateral filtering on the lamp pair", lambda: p1()),
    "p2": ("Problem 2: gradient-domain fusion on the museum pair", lambda: p2()),
    "p3-pair1": ("Problem 3, pair 1: bilateral filtering on my desk pair",
                 lambda: __import__("own_p1").main()),
    "p3-pair2": ("Problem 3, pair 2: gradient-domain fusion on my card pair",
                 lambda: __import__("own_p2").main()),
    "bonus-reflection": ("Bonus 4: reflection removal on my window pair",
                         lambda: __import__("own_bonus").main()),
    "bonus-npr": ("Bonus 5: gradient-domain non-photorealistic rendering",
                  lambda: __import__("bonus_npr").main()),
    "figures-p1": ("Report figures for Problem 1 (parameter sweeps, masks, comparisons)",
                   lambda: __import__("report_figures_p1").main()),
    "figures-p2": ("Report figures for Problem 2 (re-integration, gradients, boundary tests)",
                   lambda: __import__("report_figures_p2").main()),
}


def cli():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Computational Photography HW3: run one problem (or all).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:18s} {v[0]}" for k, v in COMMANDS.items()))
    parser.add_argument("command", choices=list(COMMANDS) + ["all"], help="problem to run")
    args = parser.parse_args()
    for name in (COMMANDS if args.command == "all" else [args.command]):
        print(f"=== {name}: {COMMANDS[name][0]}")
        COMMANDS[name][1]()


if __name__ == "__main__":
    cli()
