"""Figure of the current IPM calibration, for reports and presentations.

Left:  the calibration photo with the 4 clicked corners of the tape rectangle and its size.
Right: the same photo warped to the bird's-eye view (BEV), with 1 m distance lines.
       If the calibration is right, the tape rectangle looks like a true rectangle
       of the stated size, and straight floor joints look parallel.

Usage:
    python calib_figure.py                       # photo stored in bev_calibration.json
    python calib_figure.py --image other.jpg     # same calibration over another photo
Output: ~/Assistive-Navigation/logs/evidence/calib_figure.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import bev
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="photo to show (default: the one used for the calibration)")
    ap.add_argument("--out", default=str(C.BASE / "logs" / "evidence" / "calib_figure.jpg"))
    args = ap.parse_args()

    if not bev.CALIB_FILE.exists():
        sys.exit("No bev_calibration.json: calibrate first")
    H, data = bev.load_calibration()
    photo = args.image or data.get("photo")
    if not photo or not Path(photo).expanduser().exists():
        sys.exit(f"Calibration photo not found ({photo}). Pass one with --image")
    img = cv2.imread(str(Path(photo).expanduser()))
    if img.shape[1] != 1920:
        img = cv2.resize(img, (1920, 1080))

    # Left panel: photo + clicked corners
    left = img.copy()
    pts = data.get("points_px")
    w, l, near = data.get("rect_m", C.CALIB_RECT)
    if pts:
        p = np.array(pts, np.int32)
        cv2.polylines(left, [p], True, (0, 165, 255), 4, cv2.LINE_AA)
        for k, (u, v) in enumerate(p):
            cv2.circle(left, (int(u), int(v)), 12, (0, 0, 255), -1)
            cv2.putText(left, str(k + 1), (int(u) + 14, int(v) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 3, cv2.LINE_AA)
    # Distance lines projected on the photo (1 m steps)
    Hi = np.linalg.inv(H)
    for y in np.arange(1.0, 6.01, 1.0):
        line = []
        for x in np.linspace(-2, 2, 41):
            q = Hi @ [x, y, 1.0]
            line.append((q[0] / q[2], q[1] / q[2]))
        line = np.array(line)
        ok = (line[:, 1] > 0) & (line[:, 1] < 1080)
        if ok.sum() > 1:
            cv2.polylines(left, [line[ok].astype(np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
            u, v = line[ok][0]
            cv2.putText(left, f"{y:.0f} m", (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(left, (0, 0), (1920, 60), (0, 0, 0), -1)
    cv2.putText(left, f"Calibration photo: tape rectangle {w:.1f} x {l:.1f} m, near edge {near:.1f} m from the feet",
                (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Right panel: bird's-eye view of the photo, 1 cm per pixel for a sharper look
    res = 0.01
    top = bev.warp_to_grid(img, H, res=res)
    top = bev.draw_grid(top, 1.0, res=res)
    A = bev.ground_to_grid_matrix(res)
    if pts:
        g = np.array([[-w / 2, near], [w / 2, near], [w / 2, near + l], [-w / 2, near + l]])
        gp = (A @ np.c_[g, np.ones(4)].T).T
        cv2.polylines(top, [gp[:, :2].astype(np.int32)], True, (0, 165, 255), 3, cv2.LINE_AA)
    _, y_bottom = bev.image_to_ground(H, 960, 1079)
    yb = int(A[1, 1] * y_bottom + A[1, 2])
    top[yb:] = (top[yb:] * 0.35).astype(np.uint8)                       # blind zone
    cv2.putText(top, f"blind zone < {y_bottom:.2f} m", (12, min(top.shape[0] - 12, yb + 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(top, "Bird's-eye view (BEV)", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    s = 1080 / top.shape[0]
    top = cv2.resize(top, (int(top.shape[1] * s), 1080), interpolation=cv2.INTER_AREA)
    fig = np.hstack([left, top])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, fig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Photo: {photo}\nSaved {args.out}  ({fig.shape[1]}x{fig.shape[0]})")


if __name__ == "__main__":
    main()
