from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage import data, color, transform, util


def to_u8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = color.rgb2gray(image)
    x = np.asarray(image, dtype=np.float64)
    x = np.clip(x, 0.0, 1.0) if x.max() <= 1.5 else np.clip(x / 255.0, 0.0, 1.0)
    return np.round(255 * x).astype(np.uint8)


def transform_image(img: np.ndarray, angle: float, scale: float, gamma: float,
                    noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    center = (w / 2.0, h / 2.0)
    M = cv.getRotationMatrix2D(center, angle, scale)
    out = cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_REFLECT101)
    out = np.clip((out.astype(np.float64) / 255.0) ** gamma, 0, 1)
    rng = np.random.default_rng(seed)
    out = np.clip(out + rng.normal(0, noise_std, out.shape), 0, 1)
    return np.round(255 * out).astype(np.uint8), M


def detector(method: str):
    if method == "sift":
        return cv.SIFT_create(nfeatures=2500), cv.NORM_L2
    if method == "orb":
        return cv.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8), cv.NORM_HAMMING
    if method == "akaze":
        return cv.AKAZE_create(), cv.NORM_HAMMING
    raise ValueError(method)


def evaluate_pair(img1: np.ndarray, img2: np.ndarray, M: np.ndarray, method: str,
                  ratio: float = 0.75) -> dict[str, float | int | str]:
    det, norm = detector(method)
    t0 = time.perf_counter()
    k1, d1 = det.detectAndCompute(img1, None)
    k2, d2 = det.detectAndCompute(img2, None)
    t_detect = time.perf_counter() - t0
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return {"method": method, "n1": len(k1), "n2": len(k2), "good": 0,
                "geom_correct": 0, "precision": 0.0, "match_ms": 0.0,
                "detect_ms": 1000*t_detect}
    matcher = cv.BFMatcher(normType=norm, crossCheck=False)
    t1 = time.perf_counter()
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    t_match = time.perf_counter() - t1
    correct = 0
    errors = []
    for m in good:
        p = np.array([k1[m.queryIdx].pt[0], k1[m.queryIdx].pt[1], 1.0])
        gt = M @ p
        q = np.array(k2[m.trainIdx].pt)
        err = float(np.linalg.norm(gt - q))
        errors.append(err)
        correct += int(err < 3.0)
    return {
        "method": method,
        "n1": len(k1), "n2": len(k2), "good": len(good),
        "geom_correct": correct,
        "precision": correct / max(len(good), 1),
        "median_error": float(np.median(errors)) if errors else float("nan"),
        "detect_ms": 1000*t_detect, "match_ms": 1000*t_match,
    }


def ransac_demo(seed: int = 42) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 100)
    y = 1.8*x + 1.5 + rng.normal(0, 0.45, len(x))
    out = np.column_stack([rng.uniform(0, 10, 45), rng.uniform(-3, 22, 45)])
    pts = np.vstack([np.column_stack([x, y]), out])
    best_mask = np.zeros(len(pts), dtype=bool)
    best = (0.0, 0.0)
    for _ in range(1500):
        i, j = rng.choice(len(pts), 2, replace=False)
        dx = pts[j, 0] - pts[i, 0]
        if abs(dx) < 1e-12:
            continue
        a = (pts[j, 1]-pts[i, 1])/dx
        b = pts[i, 1]-a*pts[i, 0]
        r = np.abs(a*pts[:, 0]-pts[:, 1]+b)/np.sqrt(a*a+1)
        mask = r < 0.8
        if mask.sum() > best_mask.sum():
            best_mask, best = mask, (a, b)
    a, b = np.polyfit(pts[best_mask, 0], pts[best_mask, 1], 1)
    return {"inliers": int(best_mask.sum()), "total": len(pts),
            "estimated_slope": float(a), "estimated_intercept": float(b)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    img = to_u8_gray(data.camera())
    rows = []
    cases = [
        (0, 1.0, 1.0, 0.00),
        (25, 1.0, 1.0, 0.00),
        (0, 0.75, 1.0, 0.00),
        (35, 0.80, 0.70, 0.02),
        (-45, 1.25, 1.40, 0.04),
    ]
    for case_id, (angle, scale, gamma, noise) in enumerate(cases):
        img2, M = transform_image(img, angle, scale, gamma, noise, args.seed+case_id)
        for method in ("sift", "orb", "akaze"):
            r = evaluate_pair(img, img2, M, method)
            r.update({"case": case_id, "angle": angle, "scale": scale,
                      "gamma": gamma, "noise_std": noise})
            rows.append(r)
    with (args.output / "feature_matching_synthetic.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv.__version__,
            "numpy": np.__version__,
        },
        "ransac_demo": ransac_demo(args.seed),
        "notes": "Synthetic educational results; not a published benchmark.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
