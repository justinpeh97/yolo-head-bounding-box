"""
Run YOLO pose estimation on a folder of images and draw ALL face/head
keypoints (body keypoints ignored). Keypoint confidence is not used
anywhere - every predicted point (i.e. anything other than the (0, 0)
"not predicted at all" sentinel) is trusted as-is, both for display and
for the head bbox math.

A head bounding box is estimated in two cases, chosen GEOMETRICALLY
(nose x-position relative to both ears): "frontal" if the nose sits
between the ears, "profile" otherwise.

Requirements:
    pip install ultralytics opencv-python numpy

Note: "YOLO26" needs a recent version of the `ultralytics` package that
supports it. If model loading fails, run: pip install -U ultralytics

Usage:
    python pose_estimation.py <input_dir> <output_dir> [--model yolo26m-pose.pt]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Keypoint indices, matching Ultralytics' COCO-17 order. Body keypoints
# (5-16) are ignored entirely - only the face/head is drawn.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
HEAD_KEYPOINT_INDICES = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]

HEAD_SKELETON = [
    (LEFT_EYE, RIGHT_EYE), (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
]

POINT_COLOR = (0, 255, 0)    # BGR green
LINE_COLOR = (255, 128, 0)   # BGR blue skeleton lines
POINT_RADIUS = 4
LINE_THICKNESS = 2

HEAD_BBOX_COLOR = (255, 0, 255)  # BGR magenta
HEAD_BBOX_THICKNESS = 2

# Frontal case: how much to pad the ear-to-ear span, and how much taller
# the top half is than the bottom half (hair/skull vs. chin).
FRONTAL_PAD_FRAC = 0.25
FRONTAL_TOP_EXTRA_FRAC = 0.3

# Profile case: nose-to-ear distance, measured on whichever ear sits
# farther from the nose (the near/visible ear), is the scale reference.
# PROFILE_BACK_FRAC extends the box behind that ear (towards the occluded
# back of the skull); PROFILE_FRONT_FRAC extends it in front of the nose
# (towards lips/chin). The vertical extent is split around the mean y of
# nose + both ears: PROFILE_UP_FRAC of the box width above that point,
# PROFILE_DOWN_FRAC below it.
PROFILE_BACK_FRAC = 0.5
PROFILE_FRONT_FRAC = 0.2
PROFILE_UP_FRAC = 0.7
PROFILE_DOWN_FRAC = 0.55


def get_head_orientation(pts, valid):
    """Classify head orientation from the nose/ear x-positions.

    Returns "frontal", "profile", or None if nose/either ear wasn't
    predicted at all (can't classify).
    """
    if not (valid[NOSE] and valid[LEFT_EAR] and valid[RIGHT_EAR]):
        return None
    nose_x = pts[NOSE, 0]
    ear_x_min = min(pts[LEFT_EAR, 0], pts[RIGHT_EAR, 0])
    ear_x_max = max(pts[LEFT_EAR, 0], pts[RIGHT_EAR, 0])
    return "frontal" if ear_x_min <= nose_x <= ear_x_max else "profile"


def compute_head_bbox_frontal(pts, valid):
    """Box spans ear-to-ear (+ padding), using every valid head point."""
    idx = [i for i in HEAD_KEYPOINT_INDICES if valid[i]]
    head_pts = pts[idx]
    xmin, xmax = head_pts[:, 0].min(), head_pts[:, 0].max()
    width = xmax - xmin
    if width <= 0:
        return None

    pad = FRONTAL_PAD_FRAC * width
    xmin -= pad
    xmax += pad
    padded_width = xmax - xmin

    down = padded_width / 2
    up = down * (1 + FRONTAL_TOP_EXTRA_FRAC)

    y_center = pts[[NOSE, LEFT_EAR, RIGHT_EAR], 1].mean()
    ymin = y_center - up
    ymax = y_center + down

    return xmin, ymin, xmax, ymax


def compute_head_bbox_profile(pts, valid):
    """Extrapolate the box off the nose-to-ear distance on the ear
    farther from the nose (the near/visible ear in a real profile sits
    further from the nose than an occluded far ear typically would).
    """
    nose = pts[NOSE]
    left_ear = pts[LEFT_EAR]
    right_ear = pts[RIGHT_EAR]

    left_dist = abs(left_ear[0] - nose[0])
    right_dist = abs(right_ear[0] - nose[0])
    far_ear, far_dist = (
        (right_ear, right_dist) if right_dist > left_dist else (left_ear, left_dist)
    )

    direction = 1.0 if far_ear[0] > nose[0] else -1.0
    back_x = far_ear[0] + direction * PROFILE_BACK_FRAC * far_dist
    front_x = nose[0] - direction * PROFILE_FRONT_FRAC * far_dist
    xmin, xmax = min(front_x, back_x), max(front_x, back_x)
    width = xmax - xmin

    y_center = pts[[NOSE, LEFT_EAR, RIGHT_EAR], 1].mean()
    ymin = y_center - width * PROFILE_UP_FRAC
    ymax = y_center + width * PROFILE_DOWN_FRAC

    return xmin, ymin, xmax, ymax


def compute_head_bbox(pts, valid):
    """Classify orientation geometrically, then dispatch to the matching
    formula. Returns None if there isn't enough information for a box.
    """
    orientation = get_head_orientation(pts, valid)
    if orientation == "frontal":
        return compute_head_bbox_frontal(pts, valid)
    if orientation == "profile":
        return compute_head_bbox_profile(pts, valid)
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
            "Run YOLO pose estimation on a folder of images, draw ALL "
            "face/head keypoints, and estimate a head bbox from their "
            "geometry (no confidence weighting)."
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
