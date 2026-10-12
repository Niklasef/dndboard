"""Find miniatures on the table by diffing a photo against a photo of the empty table.

Usage: python detect.py [photo ...]   (default: images/g*.jpg)
Prints one JSON line per photo and writes annotated images to out/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGES = Path("images")
OUT = Path("out")
SCALE = 0.5  # work at half resolution
DIFF_THRESHOLD = 20  # LAB distance that counts as "something changed"
MIN_AREA = 400  # px at working scale; smaller blobs are noise
PINK = (180, 80, 255)  # BGR

# Mat corners in b1 at working scale; inset so clutter around the mat is ignored.
MAT = np.array([(380, 220), (1270, 220), (1300, 1300), (1300, 1580), (80, 1580)], np.int32)


def load(path):
    img = cv2.imread(str(path))
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def align(img, ref, mask):
    """Warp img onto ref using SIFT matches found on the mat."""
    sift = cv2.SIFT_create()
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Allow for the camera having moved a bit between shots
    wide = cv2.dilate(mask, np.ones((101, 101), np.uint8))
    k1, d1 = sift.detectAndCompute(gray(ref), mask)
    k2, d2 = sift.detectAndCompute(gray(img), wide)
    pairs = cv2.BFMatcher().knnMatch(d2, d1, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return cv2.warpPerspective(img, H, (ref.shape[1], ref.shape[0]))


def changed_mask(img, ref, mask):
    blur = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 3)
    a, b = blur(img), blur(ref)
    # Cancel overall exposure / white balance differences between the shots
    a += np.median(b[mask > 0], axis=0) - np.median(a[mask > 0], axis=0)
    diff = np.linalg.norm(a - b, axis=2)
    out = ((diff > DIFF_THRESHOLD) & (mask > 0)).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Merge the figure, base and shadow into one blob
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))


def detect(path, ref, mask):
    aligned = align(load(path), ref, mask)
    changed = changed_mask(aligned, ref, mask)
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_AREA]

    # TODO: identification (step 2). Everything is a goblin for now.
    minis = [{"label": "goblin", "bbox": [round(v / SCALE) for v in b]} for b in boxes]
    status = "none" if not boxes else "ok" if len(boxes) == 1 else f"ambiguous ({len(boxes)} blobs)"

    annotated = aligned.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), PINK, 3)
        cv2.putText(annotated, "goblin", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, PINK, 2)
    cv2.polylines(annotated, [MAT], True, (255, 255, 0), 1)
    cv2.imwrite(str(OUT / f"{path.stem}.jpg"), annotated)
    cv2.imwrite(str(OUT / f"{path.stem}_mask.jpg"), changed)
    return {"image": path.stem, "status": status, "minis": minis}


def main():
    OUT.mkdir(exist_ok=True)
    ref = load(IMAGES / "b1.jpg")
    mask = np.zeros(ref.shape[:2], np.uint8)
    cv2.fillPoly(mask, [MAT], 255)
    paths = [Path(p) for p in sys.argv[1:]] or sorted(IMAGES.glob("g*.jpg"))
    for path in paths:
        print(json.dumps(detect(path, ref, mask)))


if __name__ == "__main__":
    main()
