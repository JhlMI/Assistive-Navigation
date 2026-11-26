"""Figure of the current IPM calibration, for the thesis (texts in Spanish).

Left:  the calibration photo with the 4 clicked corners of the reference rectangle and
       the 1 m distance lines projected onto the floor.
Right: the same photo warped to the bird's-eye view (BEV), with 1 m distance lines and the
       blind zone. If the calibration is right, the rectangle looks like a true rectangle of
       the stated size and straight floor joints look parallel.

The blind zone is the same value the app uses (BevPlanner.y_start), so the figure and the
log ("blind zone X.XX m") always agree.

Usage:
    python calib_figure.py                       # photo stored in bev_calibration.json
    python calib_figure.py --image other.jpg     # same calibration over another photo
    python calib_figure.py --ref "baldosas de 0,5 m"   # how the rectangle was marked
Output: ~/Assistive-Navigation/logs/evidence/calib_figure.jpg (+ .png)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import bev
import config as C

FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def comma(x, nd=2):
    """Spanish decimal comma."""
    return f"{x:.{nd}f}".replace(".", ",")


def put_texts(img, items):
    """Draw texts with accents (PIL + DejaVu). items: (text, (x, y), size, bold, color_bgr, bg_bgr|None).
    Falls back to OpenCV (no accents) if PIL or the font is missing."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for text, (x, y), size, bold, color, bg in items:
            font = ImageFont.truetype(FONT_PATHS[1 if bold else 0], size)
            if bg is not None:
                l, t, r, b = d.textbbox((x, y), text, font=font)
                d.rectangle((l - 6, t - 4, r + 6, b + 4), fill=bg[::-1])
            d.text((x, y), text, font=font, fill=color[::-1])
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        for text, (x, y), size, bold, color, bg in items:
            ascii_text = text.encode("ascii", "replace").decode()
            cv2.putText(img, ascii_text, (int(x), int(y + size)), cv2.FONT_HERSHEY_SIMPLEX, size / 30,
                        color, 2 if bold else 1, cv2.LINE_AA)
    return img


def blind_zone_m(H):
    """Same blind zone as the app (planner), with the exact image edge as a fallback."""
    try:
        import planner
        return float(planner.BevPlanner(H).y_start)
    except Exception:
        return bev.image_to_ground(H, 960, 1079)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="photo to show (default: the one used for the calibration)")
    ap.add_argument("--ref", default="baldosas de 0,5 m", help="how the rectangle was marked (for the title)")
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
    pts = data.get("points_px")
    w, l, near = data.get("rect_m", C.CALIB_RECT)
    y_blind = blind_zone_m(H)
    Hi = np.linalg.inv(H)

    def to_img(x, y):
        q = Hi @ [x, y, 1.0]
        return q[0] / q[2], q[1] / q[2]

    # ---------------- Left panel: photo + corners + distance lines
    left = img.copy()
    texts_left = []
    for y in np.arange(2.0, 6.01, 1.0):                  # 1 m is inside the blind zone
        line = np.array([to_img(x, y) for x in np.linspace(-2, 2, 81)])
        ok = (line[:, 0] >= 0) & (line[:, 0] < 1920) & (line[:, 1] > 70) & (line[:, 1] < 1080)
        if ok.sum() > 1:
            cv2.polylines(left, [line[ok].astype(np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
            # label on the floor, 1.3 m to the left of the axis (inside the image if possible)
            u, v = to_img(-1.3, y)
            if not (0 <= u < 1800):
                u, v = line[ok][0]
            texts_left.append((f"{y:.0f} m", (int(u) + 4, int(v) - 40), 30, True, (255, 255, 255), (0, 0, 0)))
    if pts:
        p = np.array(pts, np.int32)
        cv2.polylines(left, [p], True, (0, 165, 255), 4, cv2.LINE_AA)
        for k, (u, v) in enumerate(p):
            cv2.circle(left, (int(u), int(v)), 12, (0, 0, 255), -1)
            texts_left.append((str(k + 1), (int(u) + 14, int(v) - 46), 34, True, (255, 255, 255), None))
    cv2.rectangle(left, (0, 0), (1920, 64), (0, 0, 0), -1)
    texts_left.append((f"Foto de calibración: rectángulo de {comma(w, 1)} × {comma(l, 1)} m ({args.ref}), "
                       f"borde cercano a {comma(near, 1)} m de los pies", (20, 14), 34, False, (255, 255, 255), None))
    put_texts(left, texts_left)

    # ---------------- Right panel: bird's-eye view (1 cm per pixel)
    res = 0.01
    top = bev.warp_to_grid(img, H, res=res)
    A = bev.ground_to_grid_matrix(res)
    wpx = top.shape[1]
    row = lambda y: int(A[1, 1] * y + A[1, 2])
    yb = row(y_blind)
    top[yb:] = (top[yb:] * 0.3).astype(np.uint8)                     # blind zone darkened
    for y in np.arange(1.0, 6.01, 1.0):
        cv2.line(top, (0, row(y)), (wpx, row(y)), (255, 255, 255), 1, cv2.LINE_AA)
    col0 = int(A[0, 2])
    cv2.line(top, (col0, 0), (col0, top.shape[0]), (0, 255, 255), 1, cv2.LINE_AA)   # body axis
    cv2.line(top, (0, yb), (wpx, yb), (255, 200, 0), 2, cv2.LINE_AA)               # blind zone edge
    if pts:
        g = np.array([[-w / 2, near], [w / 2, near], [w / 2, near + l], [-w / 2, near + l]])
        gp = (A @ np.c_[g, np.ones(4)].T).T
        cv2.polylines(top, [gp[:, :2].astype(np.int32)], True, (0, 165, 255), 3, cv2.LINE_AA)

    s = 1080 / top.shape[0]                                          # scale to the photo height first,
    top = cv2.resize(top, (int(wpx * s), 1080), interpolation=cv2.INTER_AREA)   # so the texts keep their size
    R = lambda y: int(row(y) * s)
    texts_right = [("Vista de pájaro (BEV)", (12, 12), 30, True, (255, 255, 255), (0, 0, 0))]
    for y in np.arange(2.0, 6.01, 1.0):
        if R(y) > 70:
            texts_right.append((f"{y:.0f} m", (8, R(y) - 34), 24, True, (255, 255, 255), (0, 0, 0)))
    texts_right.append((f"Zona ciega: {comma(y_blind)} m", (12, int(yb * s) + 16), 28, True, (255, 200, 0), None))
    put_texts(top, texts_right)

    fig = np.hstack([left, np.full((1080, 8, 3), 255, np.uint8), top])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fig, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out.with_suffix(".png")), fig)
    print(f"Photo: {photo}\nBlind zone: {y_blind:.2f} m (same as the app)\n"
          f"Saved {out} and {out.with_suffix('.png').name}  ({fig.shape[1]}x{fig.shape[0]})")


if __name__ == "__main__":
    main()
