# pose_testing

Runs a YOLO pose model over a folder of images, plots the face/head
keypoints (nose, eyes, ears - body keypoints are ignored), and draws an
estimated head bounding box for each detected person.

## Setup

```bash
pip install -r requirements.txt
```

Needs a recent `ultralytics` version that supports the YOLO26 pose
models. If the model fails to load, run `pip install -U ultralytics`.

The pose model weights (e.g. `yolo26m-pose.pt`) aren't checked into this
repo (see `.gitignore`) - Ultralytics will auto-download known model
names on first run, or point `--model` at a local weights file.

## Usage

```bash
python pose_estimation.py <input_dir> <output_dir> [options]
```

| Argument | Description |
|---|---|
| `input_dir` | Folder of images to process |
| `output_dir` | Folder to write annotated images to (created if missing) |
| `--model` | Path or name of the YOLO pose weights (default: `yolo26m-pose.pt`) |
| `--device` | Inference device, e.g. `cpu`, `0`, `0,1` (default: auto) |
| `--det-conf` | Person-detection confidence threshold (default: `0.1`) |

Example:

```bash
python pose_estimation.py ./images ./output --model yolo26m-pose.pt
```

## What gets drawn

- **Green dots** - every predicted face/head keypoint (nose, both eyes,
  both ears). All of them are trusted as-is; keypoint confidence isn't
  used anywhere, only whether the model predicted a point at all.
- **Blue lines** - a small skeleton connecting the head keypoints.
- **Magenta box** - the estimated head bounding box.

## How the head box is estimated

Orientation is classified geometrically from the nose/ear x-positions:
if the nose sits horizontally between both ears, the head is treated as
**frontal** (facing the camera or away from it); otherwise it's
**profile** (facing left or right). A different formula is used for
each case:

- **Frontal**: the box spans ear-to-ear (with padding), sized off that
  width. The bottom half of the box (toward the chin) and the top half
  (toward the hair/skull) are asymmetric - more room is given above.
- **Profile**: since only one side of the head is visible, the
  nose-to-ear distance (on whichever ear sits farther from the nose) is
  used as a scale reference, and the box is extrapolated forward past
  the nose and backward past the ear to approximate the full head.

If there isn't enough information to classify orientation or compute a
box (e.g. the nose or an ear was never predicted), only the keypoints
are drawn - no box.

## Known limitations

- Only left-right head rotation (yaw) is modeled. A head tilted sharply
  up or down (pitch) can still produce a loose or misplaced box, since
  the 5 face/head keypoints don't directly encode pitch.
- Since keypoint confidence isn't used as a filter, an occasional
  high-confidence-looking but wrong keypoint prediction (e.g. the pose
  model mistaking the back of someone's head for a face) can produce a
  bad box.
