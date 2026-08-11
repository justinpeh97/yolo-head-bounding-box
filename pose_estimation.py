"""
Run YOLO pose estimation on a folder of images and estimate a head
bounding box for every detected person, using only rule-based geometry
on the COCO-17 keypoints (no trained head detector).

The model ("ear-axis" model) rests on two empirical observations,
calibrated against ~880 ground-truth head boxes (wcp_107_face dataset):

1. CENTER: the midpoint of the two ear keypoints sits on the head's
   vertical rotation axis, so its x-coordinate is an almost unbiased
   estimate of the head-box center at EVERY yaw angle (frontal, 3/4,
   profile, even back view). No frontal/profile case split is needed
   for the center. Vertically, the box center sits slightly above the
   ear midpoint.

2. SIZE: the distance from the nose to the FARTHER ear is a remarkably
   yaw-stable fraction of the head-box width (~0.44 frontal, peaking
   ~0.54 at three-quarter views, back to ~0.44 in profile). A small
   piecewise-linear correction g(s) over the frontality ratio s removes
   that mid-yaw hump. As a floor, the box is never narrower than
   EAR_DIST_FLOOR x the ear-to-ear distance (which dominates for
   frontal faces). Height is a fixed aspect multiple of width.

The frontality ratio s = |nose_x - mid_ear_x| / (ear_x_span / 2) is 0
for a perfectly frontal face and grows past ~2 for full profiles.

Keypoint confidence is not used; any keypoint other than the (0, 0)
"not predicted" sentinel is trusted as-is. Fallbacks cover partial
keypoint sets (ears without nose = back view; eyes only; shoulders
only), so a box is produced whenever geometry allows.

Measured against ground truth (178 images, 901 heads), versus the
frontal/profile method in pose_estimation.py:
    mean IoU over all GT heads:  0.645 -> 0.718
    recall @ IoU >= 0.5:         0.851 -> 0.919
(gains confirmed on held-out split halves).

Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    python head_bbox_estimation.py <input_dir> <output_dir> [--model yolo26m-pose.pt]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Keypoint indices, matching Ultralytics' COCO-17 order.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
HEAD_KEYPOINT_INDICES = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]

HEAD_SKELETON = [
    (LEFT_EYE, RIGHT_EYE), (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
]

POINT_COLOR = (0, 255, 0)    # BGR green
LINE_COLOR = (255, 128, 0)   # BGR blue skeleton lines
POINT_RADIUS = 4
LINE_THICKNESS = 2

HEAD_BBOX_COLOR = (0, 215, 255)  # BGR gold (distinct from the magenta
HEAD_BBOX_THICKNESS = 2          # boxes drawn by pose_estimation.py)

# ---------------------------------------------------------------------------
# Calibrated constants (coordinate-descent fit on GT head boxes; values are
# within noise of the raw medians measured from the data, see module docstring)

# g(s): nose-to-far-ear distance as a fraction of head-box width, as a
# piecewise-linear function of the frontality ratio s.
G_FRONTAL = 0.44      # g at s = 0
G_MID = 0.54          # g at s = S_MID (three-quarter view)
G_PROFILE = 0.44      # g at s >= S_HI (full profile and beyond)
S_MID = 0.6
S_HI = 4.0

# Box width is never below this multiple of the ear-to-ear distance
# (the binding constraint for frontal faces).
EAR_DIST_FLOOR = 1.55

ASPECT = 1.16         # box height / width
CENTER_UP_FRAC = 0.04  # center sits this fraction of box height above mid-ear
CENTER_BACK_FRAC = 0.02  # tiny center shift from nose toward mid-ear

# Fallback when the pose model predicts no usable face keypoints at all:
# head size and position from the shoulders.
SHOULDER_WIDTH_FRAC = 0.42  # box width as fraction of shoulder span
SHOULDER_UP_FRAC = 0.55     # head center this fraction of span above shoulders


def frontality_ratio(pts):
    """s = |nose_x - mid_ear_x| / (ear_x_span / 2).

    0 for a perfectly frontal face, ~1 with the nose over an ear,
    >2 for full profile views.
    """
    ear_span = abs(pts[LEFT_EAR, 0] - pts[RIGHT_EAR, 0])
    mid_x = (pts[LEFT_EAR, 0] + pts[RIGHT_EAR, 0]) / 2.0
    return abs(pts[NOSE, 0] - mid_x) / max(ear_span / 2.0, 1e-6)


def yaw_width_ratio(s):
    """g(s): piecewise-linear nose-to-far-ear / box-width ratio."""
    if s <= S_MID:
        return G_FRONTAL + (G_MID - G_FRONTAL) * (s / S_MID)
    if s <= S_HI:
        return G_MID + (G_PROFILE - G_MID) * ((s - S_MID) / (S_HI - S_MID))
    return G_PROFILE


def compute_head_bbox(pts, valid):
    """Estimate (xmin, ymin, xmax, ymax) for one person's head, or None.

    Primary path needs nose + both ears (the pose model virtually always
    predicts all five face points, even for back-of-head views).
    """
    if not (valid[NOSE] and valid[LEFT_EAR] and valid[RIGHT_EAR]):
        return compute_head_bbox_fallback(pts, valid)

    nose = pts[NOSE]
    left_ear, right_ear = pts[LEFT_EAR], pts[RIGHT_EAR]
    mid_ear = (left_ear + right_ear) / 2.0
    ear_dist = float(np.linalg.norm(left_ear - right_ear))
    far = max(
        float(np.linalg.norm(nose - left_ear)),
        float(np.linalg.norm(nose - right_ear)),
    )
    if far < 1e-6 and ear_dist < 1e-6:
        return None

    s = frontality_ratio(pts)
    width = max(far / yaw_width_ratio(s), EAR_DIST_FLOOR * ear_dist)
    height = ASPECT * width

    direction = 1.0 if mid_ear[0] >= nose[0] else -1.0
    cx = mid_ear[0] + direction * CENTER_BACK_FRAC * width
    cy = mid_ear[1] - CENTER_UP_FRAC * height
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def compute_head_bbox_fallback(pts, valid):
    """Partial keypoint sets: ears without nose (back view), a couple of
    face points (occlusions), or shoulders only."""
    face_idx = [i for i in HEAD_KEYPOINT_INDICES if valid[i]]
    if len(face_idx) >= 2:
        face_pts = pts[face_idx]
        diffs = face_pts[:, None, :] - face_pts[None, :, :]
        scale = float(np.sqrt((diffs ** 2).sum(-1)).max())
        if scale > 1e-6:
            if valid[LEFT_EAR] and valid[RIGHT_EAR]:
                # Back of head: ear distance ~ head width.
                width = EAR_DIST_FLOOR * scale
                center = (pts[LEFT_EAR] + pts[RIGHT_EAR]) / 2.0
            else:
                # Eyes/nose only: treat the largest pairwise distance
                # like a frontal nose-to-ear span.
                width = scale / G_FRONTAL
                center = face_pts.mean(axis=0)
            height = ASPECT * width
            cy = center[1] - CENTER_UP_FRAC * height
            return (center[0] - width / 2, cy - height / 2,
                    center[0] + width / 2, cy + height / 2)

    if valid[LEFT_SHOULDER] and valid[RIGHT_SHOULDER]:
        span = float(np.linalg.norm(pts[LEFT_SHOULDER] - pts[RIGHT_SHOULDER]))
        if span > 1e-6:
            width = SHOULDER_WIDTH_FRAC * span
            height = ASPECT * width
            cx = (pts[LEFT_SHOULDER, 0] + pts[RIGHT_SHOULDER, 0]) / 2.0
            cy = (pts[LEFT_SHOULDER, 1] + pts[RIGHT_SHOULDER, 1]) / 2.0 \
                - SHOULDER_UP_FRAC * span
            return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    return None


def draw_person(image, pts):
    """Draw one person's head keypoints/skeleton and (if possible) head bbox."""
    valid = ~np.all(pts == 0, axis=1)

    for i, j in HEAD_SKELETON:
        if not (valid[i] and valid[j]):
            continue
        pt1 = tuple(pts[i].astype(int))
        pt2 = tuple(pts[j].astype(int))
        cv2.line(image, pt1, pt2, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

    for k in HEAD_KEYPOINT_INDICES:
        if not valid[k]:
            continue
        x, y = pts[k].astype(int)
        cv2.circle(image, (x, y), POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)

    head_bbox = compute_head_bbox(pts, valid)
    if head_bbox is not None:
        xmin, ymin, xmax, ymax = (int(round(v)) for v in head_bbox)
        cv2.rectangle(
            image, (xmin, ymin), (xmax, ymax),
            HEAD_BBOX_COLOR, HEAD_BBOX_THICKNESS, cv2.LINE_AA,
        )

    return image


def draw_all_keypoints(image, keypoints_xy):
    num_people = keypoints_xy.shape[0]
    for person_idx in range(num_people):
        pts = keypoints_xy[person_idx]  # (num_kpts, 2)
        draw_person(image, pts)
    return image


def process_folder(input_dir, output_dir, model_path, device=None, det_conf=0.25):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for img_path in image_paths:
        results = model.predict(
            source=str(img_path), conf=det_conf, device=device, verbose=False
        )
        result = results[0]
        image = result.orig_img.copy()

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            image = draw_all_keypoints(image, keypoints_xy)
        else:
            print(f"No people detected in {img_path.name}")

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), image)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO pose estimation on a folder of images and draw a "
            "geometry-derived head bounding box for every detected person "
            "(ear-axis model; see module docstring)."
        )
    )
    parser.add_argument("input_dir", type=str, help="Folder containing input images")
    parser.add_argument("output_dir", type=str, help="Folder to save annotated images")
    parser.add_argument(
        "--model", type=str, default="yolo26m-pose.pt",
        help="Path or name of the YOLO pose model weights (default: yolo26m-pose.pt)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference, e.g. 'cpu', '0', '0,1' (default: auto)",
    )
    parser.add_argument(
        "--det-conf", type=float, default=0.1,
        help="Confidence threshold for detecting a PERSON in the image",
    )
    args = parser.parse_args()

    process_folder(
        args.input_dir, args.output_dir, args.model,
        device=args.device, det_conf=args.det_conf,
    )


if __name__ == "__main__":
    main()

