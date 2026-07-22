#!/usr/bin/env python3
"""Reproducible experiments for Chapter 8: Image Segmentation.

The script creates a synthetic but non-trivial scene, evaluates global/local
thresholding and region-based segmentation, and writes images and CSV/JSON
summaries. It does not download data and is deterministic for a fixed seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage as ndi
import skimage
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.measure import label
from skimage.segmentation import chan_vese, random_walker, slic


def make_scene(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    x = xx / (n - 1)
    y = yy / (n - 1)

    gt = (
        ((x - 0.32) ** 2 / 0.15**2 + (y - 0.48) ** 2 / 0.22**2 <= 1)
        | ((x > 0.56) & (x < 0.83) & (y > 0.24) & (y < 0.73))
    )
    small = (x - 0.75) ** 2 + (y - 0.82) ** 2 <= 0.045**2
    gt |= small

    illumination = 0.18 + 0.48 * x + 0.10 * np.sin(2 * np.pi * y)
    texture = 0.035 * np.sin(17 * np.pi * x) * np.cos(11 * np.pi * y)
    clean = illumination + texture + 0.35 * gt.astype(float)
    clean = np.clip(clean, 0, 1)

    sigma_map = 0.025 + 0.045 * y
    noisy = clean + rng.normal(0, sigma_map)
    # Sparse impulsive contamination.
    impulse = rng.random((n, n)) < 0.006
    noisy[impulse] = rng.choice([0.0, 1.0], size=impulse.sum())
    return np.clip(noisy, 0, 1), gt


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = confusion(pred, gt)
    eps = np.finfo(float).eps
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "BalancedAccuracy": float(balanced_accuracy),
    }


def run_method(name: str, fn, image: np.ndarray, gt: np.ndarray):
    tic = perf_counter()
    pred = fn(image).astype(bool)
    elapsed_ms = 1000.0 * (perf_counter() - tic)
    result = metrics(pred, gt)
    result.update({"Method": name, "Time_ms": elapsed_ms})
    return pred, result


def otsu_method(image: np.ndarray) -> np.ndarray:
    return image > threshold_otsu(image)


def sauvola_method(image: np.ndarray) -> np.ndarray:
    t = threshold_sauvola(image, window_size=41, k=0.16)
    return image > t


def random_walker_method(image: np.ndarray) -> np.ndarray:
    # Markers are generated from conservative intensity quantiles; unknown=0.
    markers = np.zeros_like(image, dtype=np.uint8)
    q_low, q_high = np.quantile(image, [0.30, 0.78])
    markers[image < q_low] = 1
    markers[image > q_high] = 2
    labels = random_walker(image, markers, beta=60, mode="bf")
    return labels == 2


def chan_vese_method(image: np.ndarray) -> np.ndarray:
    return chan_vese(
        image,
        mu=0.18,
        lambda1=1.0,
        lambda2=1.0,
        tol=1e-3,
        max_num_iter=180,
        dt=0.5,
        init_level_set="checkerboard",
        extended_output=False,
    )


def postprocess(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=2)
    lab = label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= 35
    keep[0] = False
    return keep[lab]


def save_overview(image: np.ndarray, gt: np.ndarray, predictions: dict[str, np.ndarray], out: Path) -> None:
    ncols = 3
    panels = [("Noisy image", image), ("Ground truth", gt)] + list(predictions.items())
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.savefig(out / "segmentation_overview.png", dpi=180)
    fig.savefig(out / "segmentation_overview.pdf")
    plt.close(fig)


def save_superpixels(image: np.ndarray, out: Path) -> None:
    rgb = np.repeat(image[..., None], 3, axis=2)
    labels = slic(rgb, n_segments=180, compactness=12, sigma=1, start_label=1, channel_axis=-1)
    boundaries = ndi.morphological_gradient(labels, size=(3, 3)) > 0
    vis = exposure.rescale_intensity(rgb, out_range=(0, 1))
    vis[boundaries] = (1, 0, 0)
    plt.figure(figsize=(5.5, 5.5))
    plt.imshow(vis)
    plt.title("SLIC superpixels")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "slic_superpixels.png", dpi=180)
    plt.savefig(out / "slic_superpixels.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("chapter08_results"))
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image, gt = make_scene(args.size, args.seed)
    methods = {
        "Otsu": otsu_method,
        "Sauvola": sauvola_method,
        "Random walker": random_walker_method,
        "Chan--Vese": chan_vese_method,
    }
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, float | str]] = []
    for name, fn in methods.items():
        pred, row = run_method(name, fn, image, gt)
        predictions[name] = pred
        rows.append(row)
        pp = postprocess(pred)
        predictions[f"{name} + morphology"] = pp
        pp_row = metrics(pp, gt)
        pp_row.update({"Method": f"{name} + morphology", "Time_ms": float("nan")})
        rows.append(pp_row)

    fields = ["Method", "IoU", "Dice", "Precision", "Recall", "BalancedAccuracy", "Time_ms"]
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    save_overview(image, gt, predictions, args.output)
    save_superpixels(image, args.output)
    np.savez_compressed(args.output / "synthetic_scene.npz", image=image, ground_truth=gt)

    metadata = {
        "seed": args.seed,
        "size": args.size,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "note": "Synthetic benchmark; do not interpret it as a substitute for real-data validation.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
