"""Find miniatures on the table and label them by character.

Diffs a photo against a photo of the empty table, then matches each blob against
reference photos of every character (one folder per character under images/).

Usage: python detect.py [photo ...]   (default: images/mixed/*.jpg)
Prints one JSON line per photo and writes annotated images to out/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGES = Path("images")
OUT = Path("out")
SCALE = 0.5  # detection works at half resolution
DIFF_THRESHOLD = 20  # LAB distance that counts as "something changed"
MIN_AREA = 1500  # px at working scale; smaller blobs are noise
MIN_SIM = 0.3  # cosine similarity the best character needs before we trust a label
MIN_GAP = 0.05  # best character must beat the runner-up by this much
PINK = (180, 80, 255)  # BGR
YELLOW = (0, 220, 255)

# Mat area at working scale; inset so keyboard, mouse, cable and mat edges are ignored.
MAT = np.array([(400, 60), (1110, 60), (1110, 1060), (240, 1060)], np.int32)

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(path):
    """Return the photo at full resolution and at working scale."""
    full = cv2.imread(str(path))
    return full, cv2.resize(full, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def homography(img, ref, mask):
    """Transform that maps img onto ref, from SIFT matches found on the mat."""
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Allow for the camera having moved a bit between shots
    wide = cv2.dilate(mask, np.ones((101, 101), np.uint8))
    k1, d1 = sift.detectAndCompute(gray(ref), mask)
    k2, d2 = sift.detectAndCompute(gray(img), wide)
    pairs = matcher.knnMatch(d2, d1, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H


def changed_mask(img, ref, mask):
    blur = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 3)
    a, b = blur(img), blur(ref)
    # Cancel overall exposure / white balance differences between the shots
    a += np.median(b[mask > 0], axis=0) - np.median(a[mask > 0], axis=0)
    diff = np.linalg.norm(a - b, axis=2)
    out = ((diff > DIFF_THRESHOLD) & (mask > 0)).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Merge the figure, base and shadow into one blob without gluing neighbours together
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def find_minis(path, ref, mask):
    """Return the aligned half-res photo, the changed mask, bounding boxes and an embedding per box."""
    full, small = load(path)
    H = homography(small, ref, mask)
    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(small, H, (w, h))
    changed = changed_mask(aligned, ref, mask)

    # Identification needs more detail than detection, so warp the full-res photo too:
    # H maps half-res coords, so conjugate it by the scale change.
    S = np.diag([1 / SCALE, 1 / SCALE, 1.0])
    full_aligned = cv2.warpPerspective(full, S @ H @ np.linalg.inv(S), (round(w / SCALE), round(h / SCALE)))
    full_changed = cv2.resize(changed, None, fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_AREA]
    feats = [embed(full_aligned, full_changed, b) for b in boxes]
    return aligned, changed, boxes, feats


def embed(img, changed, box):
    """DINOv2 embedding of just the miniature, on a neutral background."""
    x, y, w, h = [round(v / SCALE) for v in box]
    crop = img[y:y + h, x:x + w].copy()
    # Only look at the miniature itself; the mat texture would otherwise sway the result
    crop[changed[y:y + h, x:x + w] == 0] = 128
    side = max(w, h)
    square = cv2.copyMakeBorder(crop, (side - h) // 2, side - h - (side - h) // 2,
                                (side - w) // 2, side - w - (side - w) // 2,
                                cv2.BORDER_CONSTANT, value=(128, 128, 128))
    rgb = cv2.cvtColor(cv2.resize(square, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(((rgb / 255.0 - MEAN) / STD).astype(np.float32)).permute(2, 0, 1)[None]
    with torch.no_grad():
        v = model(tensor)[0]
    return (v / v.norm()).numpy()


def center(v, mu):
    """All miniatures are dark grey, so their embeddings share a big common part; remove it."""
    v = v - mu
    return v / np.linalg.norm(v)


def load_refs(ref, mask):
    """Embeddings of each reference photo, keyed by character (the folder name), and their mean."""
    refs = {}
    for folder in sorted(p for p in IMAGES.iterdir() if p.is_dir() and p.name != "mixed"):
        refs[folder.name] = []
        for path in sorted(folder.glob("*.jpg")):
            _, _, boxes, feats = find_minis(path, ref, mask)
            if boxes:
                biggest = max(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3])
                refs[folder.name].append(feats[biggest])
    mu = np.mean([r for rs in refs.values() for r in rs], axis=0)
    return {name: [center(r, mu) for r in rs] for name, rs in refs.items()}, mu


def identify(f, refs):
    scores = {name: max(float(f @ r) for r in rs) for name, rs in refs.items()}
    (best, top), (_, second) = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    sure = top >= MIN_SIM and top - second >= MIN_GAP
    return (best if sure else "unknown"), scores


def detect(path, ref, mask, refs, mu):
    aligned, changed, boxes, feats = find_minis(path, ref, mask)
    minis = []
    annotated = aligned.copy()
    for box, f in zip(boxes, feats):
        label, scores = identify(center(f, mu), refs)
        minis.append({"label": label, "bbox": [round(v / SCALE) for v in box], "scores": {k: round(v, 3) for k, v in scores.items()}})
        x, y, w, h = box
        color = YELLOW if label == "unknown" else PINK
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.polylines(annotated, [MAT], True, (255, 255, 0), 1)
    cv2.imwrite(str(OUT / f"{path.stem}.jpg"), annotated)
    cv2.imwrite(str(OUT / f"{path.stem}_mask.jpg"), changed)

    if not minis:
        status = "none"
    elif any(m["label"] == "unknown" for m in minis):
        status = "uncertain"
    else:
        status = "ok"
    return {"image": path.stem, "status": status, "minis": minis}


def main():
    OUT.mkdir(exist_ok=True)
    _, ref = load(IMAGES / "empty-board.jpg")
    mask = np.zeros(ref.shape[:2], np.uint8)
    cv2.fillPoly(mask, [MAT], 255)
    refs, mu = load_refs(ref, mask)
    paths = [Path(p) for p in sys.argv[1:]] or sorted((IMAGES / "mixed").glob("*.jpg"))
    for path in paths:
        print(json.dumps(detect(path, ref, mask, refs, mu)))


if __name__ == "__main__":
    main()
