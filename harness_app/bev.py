"""Bird's-eye view (inverse perspective mapping) helpers.

Ground frame used everywhere:
    x = lateral distance in meters (right is positive, 0 = center of the body)
    y = forward distance in meters from the user's feet

The calibration file stores H_img2ground: a 3x3 homography from full-resolution
image pixels (1920x1080) to ground meters (x, y). It is valid only for the camera
mounting it was measured with (height and tilt on the harness).
"""
import json
from pathlib import Path

import cv2
import numpy as np


CALIB_FILE = Path(__file__).with_name("bev_calibration.json")
RUNS_FILE = Path(__file__).with_name("bev_calibration_runs.jsonl")   # every calibration, for comparison

# Default grid: 4 m wide (-2..+2), from 0 to 6 m ahead, 5 cm cells
GRID_X = (-2.0, 2.0)
GRID_Y = (0.0, 6.0)
GRID_RES = 0.05


def homography_from_points(img_pts, width_m, length_m, near_m):
    """img_pts: 4 image points (full res) clicked in order
    near-left, near-right, far-right, far-left of a rectangle on the floor,
    centered on the body, starting near_m in front of the feet."""
    w = width_m / 2.0
    ground = np.float32([[-w, near_m], [w, near_m], [w, near_m + length_m], [-w, near_m + length_m]])
    return cv2.getPerspectiveTransform(np.float32(img_pts), ground)


def save_calibration(H, extra):
    data = {"H_img2ground": np.asarray(H).tolist(), "image_size": [1920, 1080], **extra}
    CALIB_FILE.write_text(json.dumps(data, indent=2))


def append_run(record):
    with open(RUNS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_runs(path=None):
    path = Path(path or RUNS_FILE)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_calibration(path=None):
    data = json.loads(Path(path or CALIB_FILE).read_text())
    return np.array(data["H_img2ground"], dtype=np.float64), data


def image_to_ground(H, u, v):
    """Project one image pixel (full res) onto the ground. Returns (x, y) in meters."""
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def ground_to_grid_matrix(res=GRID_RES, gx=GRID_X, gy=GRID_Y):
    """Matrix that maps ground meters (x, y) to grid pixels (col, row); row 0 = farthest."""
    return np.array([[1.0 / res, 0.0, -gx[0] / res],
                     [0.0, -1.0 / res, gy[1] / res],
                     [0.0, 0.0, 1.0]])


def grid_size(res=GRID_RES, gx=GRID_X, gy=GRID_Y):
    return int(round((gx[1] - gx[0]) / res)), int(round((gy[1] - gy[0]) / res))   # (width, height)


def warp_to_grid(img, H, src_size=(1920, 1080), interp=cv2.INTER_LINEAR, res=GRID_RES):
    """Warp an image or mask to the bird's-eye grid.
    src_size: size the image corresponds to; if img is smaller (e.g. the 896x504 mask),
    it is scaled up to full-res coordinates first."""
    h, w = img.shape[:2]
    S = np.diag([src_size[0] / w, src_size[1] / h, 1.0])      # img px -> full-res px
    M = ground_to_grid_matrix(res) @ H @ S
    return cv2.warpPerspective(img, M, grid_size(res), flags=interp, borderValue=0)


def draw_grid(bev, step_m=1.0, res=GRID_RES):
    """Draw distance lines every step_m on a BGR bird's-eye image."""
    out = bev.copy()
    A = ground_to_grid_matrix(res)
    wpx, hpx = grid_size(res)
    for y in np.arange(GRID_Y[0], GRID_Y[1] + 1e-6, step_m):
        row = int(A[1, 1] * y + A[1, 2])
        cv2.line(out, (0, row), (wpx, row), (255, 255, 255), 1)
        cv2.putText(out, f"{y:.0f} m", (3, max(12, row - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    col0 = int(A[0, 2])
    cv2.line(out, (col0, 0), (col0, hpx), (0, 255, 255), 1)       # body center line
    return out
