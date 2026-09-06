#Tag-region crop: find the pale label rectangles and crop tightly to them.



from __future__ import annotations

from pathlib import Path

CROP_MAX_DIM = 1600
WORK_SIZE = 1200
PAD_FRAC = 0.35
MAX_EXTRA_HEIGHT_FRAC = 0.55


EQUIPMENT_COLUMN_FRAC = 0.27

WIDE_LEFT_FRAC = 0.16


MIN_AREA_FRAC = 0.12
MIN_WIDTH_FRAC = 0.24
MAX_X_FRAC = 0.45


def detect_tag_region(img_bgr) -> tuple[int, int, int, int, str]:
    """Detect the pale label rectangles; return (x, y, w, h, method) in full-res coords."""
    import cv2
    import numpy as np

    height, width = img_bgr.shape[:2]
    scale = WORK_SIZE / max(height, width)
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (60, 90, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_h, small_w = small.shape[:2]
    area_img = small_h * small_w
    boxes = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        aspect = box_w / max(box_h, 1)
        if 0.0005 * area_img < area < 0.05 * area_img and 1.1 < aspect < 7:
            boxes.append((x, y, box_w, box_h))
    boxes = [b for b in boxes if not (b[0] + b[2] < EQUIPMENT_COLUMN_FRAC * small_w)]

    if not boxes:
        x0 = int(width * EQUIPMENT_COLUMN_FRAC)
        return x0, 0, width - x0, height, "fallback_right"

    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes)
    ys1 = max(b[1] + b[3] for b in boxes)
    pad_x = int((xs1 - xs0) * PAD_FRAC) + 20
    pad_y = int((ys1 - ys0) * PAD_FRAC) + 20
    x0 = max(0, int((xs0 - pad_x) / scale))
    y0 = max(0, int((ys0 - pad_y) / scale))
    x1 = min(width, int((xs1 + pad_x) / scale))
    y1 = min(height, int(ys1 / scale) + int(height * MAX_EXTRA_HEIGHT_FRAC))
    if x1 <= x0 or y1 <= y0:
        x0 = int(width * EQUIPMENT_COLUMN_FRAC)
        return x0, 0, width - x0, height, "fallback_invalid"
    return x0, y0, x1 - x0, y1 - y0, "detected"


def is_suspicious(x: int, crop_w: int, crop_h: int, img_w: int, img_h: int) -> bool:
    """True when a detected box is too small, too narrow, or too far right to be real."""
    area_frac = (crop_w * crop_h) / max(img_w * img_h, 1)
    return (
        area_frac < MIN_AREA_FRAC
        or crop_w / max(img_w, 1) < MIN_WIDTH_FRAC
        or x / max(img_w, 1) > MAX_X_FRAC
    )


def crop_tag_region(image_path: str | Path, max_dim: int = CROP_MAX_DIM):
    """Crop to the detected tag region, with the wide fallback. Returns (PIL RGB, method)."""
    import cv2
    from PIL import Image

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"could not read {image_path}")
    img_h, img_w = img_bgr.shape[:2]

    x, y, crop_w, crop_h, method = detect_tag_region(img_bgr)
    if is_suspicious(x, crop_w, crop_h, img_w, img_h):
        x = int(img_w * WIDE_LEFT_FRAC)
        y, crop_w, crop_h = 0, img_w - x, img_h
        method = f"{method}_to_wide"

    crop = img_bgr[y:y + crop_h, x:x + crop_w]
    if max(crop.shape[:2]) > max_dim:
        scale = max_dim / max(crop.shape[:2])
        crop = cv2.resize(
            crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), method
