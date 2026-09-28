"""Generate a realistic healthy green rice leaf image for the simulation notebook."""
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def make_healthy_leaf(path: Path, width: int = 620, height: int = 420) -> None:
    rng = np.random.default_rng(7)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Soft green field background
    for y in range(height):
        for x in range(width):
            r, g, b = colorsys.hsv_to_rgb(0.22 + y / height * 0.04, 0.45, 0.32)
            img[y, x] = (int(r * 255), int(g * 255), int(b * 255))

    ys, xs = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    leaf = (((ys - cy) / 78.0) ** 2 + ((xs - cx) / 245.0) ** 2) <= 1.0

    # Healthy rice leaf: hue ~62-78 deg, strong saturation, parallel vein texture
    for y in range(height):
        for x in range(width):
            if not leaf[y, x]:
                continue
            h = 0.19 + 0.015 * np.sin(x / 22.0) + 0.008 * np.sin(y / 9.0) + rng.normal(0, 0.004)
            s = 0.62 + 0.05 * np.sin(x / 16.0) + rng.normal(0, 0.015)
            v = 0.58 + 0.04 * np.sin(y / 11.0) + rng.normal(0, 0.012)
            h = float(np.clip(h, 0.17, 0.22))   # ~61-79 deg
            s = float(np.clip(s, 0.48, 0.78))
            v = float(np.clip(v, 0.42, 0.72))
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            img[y, x] = (int(r * 255), int(g * 255), int(b * 255))

    out = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=0.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=95)
    print(f"Saved {path} ({width}x{height})")


if __name__ == "__main__":
    make_healthy_leaf(Path(__file__).parent / "assets" / "healthy_leaf.jpg")
