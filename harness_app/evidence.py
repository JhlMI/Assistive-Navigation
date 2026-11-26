"""Evidence generator: runs the full pipeline on still images (or frames taken from a video)
and saves one annotated figure per image + a CSV + an HTML summary for reports.

Usage:
    python evidence.py ~/Assistive-Navigation/logs/navigation/snapshots/*.jpg
    python evidence.py photos_folder/
    python evidence.py --video ~/Assistive-Navigation/logs/navigation/rec_X.avi --every 2
    python evidence.py --centroid img1.jpg img2.jpg      # force the centroid decision
    python evidence.py --lang en photos/                 # English labels (default: Spanish, for the thesis)

Output: ~/Assistive-Navigation/logs/evidence/<date_time>/
    <name>_result.png   camera overlay + bird's-eye panel (top), original + class mask (bottom)
    results.csv         per-image numbers (times, command, angle, distances)
    summary.html        all figures with their numbers, opens in any browser offline
"""
import argparse
import csv
import html
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if "ultralytics" in sys.modules:
    sys.exit("ultralytics is imported: it drops OpenCV to 1 thread. Aborting.")

import cv2
import numpy as np

import config as C

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SETTLE_FRAMES = 12                 # repeats per still image so the temporal filters settle
CLASS_NAMES = ["background", "building", "pole", "road", "sidewalk", "vegetation"]

# ---- Spanish labels for the thesis figures (--lang es, default) ------------------------------
ES_CLASSES = ["fondo", "edificio", "poste", "calzada", "acera", "vegetación"]
ES_COMMANDS = {"F": "ADELANTE", "L": "CORREGIR A LA IZQUIERDA", "R": "CORREGIR A LA DERECHA",
               "X": "DETENERSE (obstáculo)", "W": "PRECAUCIÓN (sin camino)", "E": "FALLA", "S": "SILENCIO"}
ES_OBJECTS = {"person": "persona", "bicycle": "bicicleta", "car": "auto", "motorcycle": "motocicleta",
              "bus": "bus", "truck": "camión"}
ES_REASONS = {"on path": "en la ruta", "path to the left": "ruta hacia la izquierda",
              "path to the right": "ruta hacia la derecha", "no walkable path": "sin camino transitable",
              "path lost (brief)": "ruta perdida (breve)", "obstacle": "obstáculo", "start": "inicio"}
FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LANG = "es"


def tr_reason(reason):
    if LANG != "es":
        return reason
    m = re.match(r"obstacle at ([\d.]+) m", reason or "")
    if m:
        return f"obstáculo a {float(m.group(1)):.1f} m".replace(".", ",")
    return ES_REASONS.get(reason, reason)


def pil_text(img, text, org, scale, color, thickness):
    """cv2.putText replacement with accents (DejaVu). org = baseline-left, like OpenCV."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        size = max(10, int(scale * 34))
        font = ImageFont.truetype(FONT_BOLD if thickness >= 2 else FONT_FILE, size)
        x, y = org
        # only convert the strip that holds the text (fast enough offline)
        y0, y1 = max(0, y - size - 4), min(img.shape[0], y + size // 2)
        if y1 <= y0:
            return
        strip = Image.fromarray(cv2.cvtColor(np.ascontiguousarray(img[y0:y1]), cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(strip).text((x, y - y0 - int(size * 0.85)), text, font=font, fill=tuple(int(c) for c in color[::-1]))
        img[y0:y1] = cv2.cvtColor(np.array(strip), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text.encode("ascii", "replace").decode(), org, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)


def put(img, text, org, scale, color, thickness=1):
    if LANG == "es":
        pil_text(img, text, org, scale, color, thickness)
    else:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
CLASS_COLORS = np.array([[40, 40, 40], [70, 70, 160], [0, 220, 255], [128, 64, 128],
                         [60, 200, 60], [35, 120, 35]], np.uint8)     # BGR


def collect_images(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT and p.exists():
            out.append(p)
        else:
            print(f"Skipped (not an image): {p}")
    return out


def frames_from_video(path, every_s):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, int(round(every_s * fps)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield f"{Path(path).stem}_f{i:05d}", frame
        i += 1
    cap.release()


def class_mask_image(mask, size):
    """Colored segmentation (all classes) with a legend."""
    img = CLASS_COLORS[np.clip(mask, 0, len(CLASS_COLORS) - 1)]
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    names = ES_CLASSES if LANG == "es" else CLASS_NAMES
    for i, name in enumerate(names):
        y = 22 + 22 * i
        cv2.rectangle(img, (10, y - 13), (26, y + 3), CLASS_COLORS[i].tolist(), -1)
        cv2.rectangle(img, (10, y - 13), (26, y + 3), (255, 255, 255), 1)
        put(img, name, (32, y), 0.5, (255, 255, 255), 1)
    return img


def info_bar(text_lines, width):
    bar = np.full((26 * len(text_lines) + 10, width, 3), 20, np.uint8)
    for i, t in enumerate(text_lines):
        put(bar, t, (10, 24 + 26 * i), 0.6, (235, 235, 235), 1)
    return bar


def compose(frame, walkable, boxes, res, mask, lines):
    import web
    C.WEB_WIDTH, C.WEB_HEIGHT = 1280, 720            # bigger than the live view, for reports
    if LANG == "es":
        web.put_text = pil_text
        web.CMD_NAME = ES_COMMANDS
        C.OBSTACLE_CLASSES = {k: ES_OBJECTS.get(v, v) for k, v in C.OBSTACLE_CLASSES.items()}
        reason = res.reason
        res.reason = tr_reason(reason)
    top = web.annotate(frame, walkable, boxes, res)  # camera overlay (+ BEV panel in BEV mode)
    if LANG == "es":
        res.reason = reason
    w = top.shape[1]
    half = (w // 2, int(w // 2 * 9 / 16))
    original = cv2.resize(frame, half, interpolation=cv2.INTER_AREA)
    seg = class_mask_image(mask, (w - half[0], half[1]))
    bottom = np.hstack([original, seg])
    return np.vstack([top, bottom, info_bar(lines, w)])


def write_html(out_dir, rows):
    cards = []
    for r in rows:
        stats = " · ".join(f"{k}: <b>{html.escape(str(r[k]))}</b>" for k in
                           ("command", "reason", "angle_deg", "path_len_m", "obstacle_m",
                            "sidewalk_pct", "seg_ms", "det_ms", "plan_ms"))
        cards.append(f"<figure><img src='{html.escape(r['figure'])}' loading='lazy'>"
                     f"<figcaption><b>{html.escape(r['image'])}</b><br>{stats}</figcaption></figure>")
    page = ("<!doctype html><meta charset='utf-8'><title>Evidence</title>"
            "<style>body{font-family:sans-serif;background:#111;color:#ddd;margin:16px}"
            "figure{margin:0 0 28px}img{width:100%;max-width:1400px;display:block}"
            "figcaption{font-size:14px;margin-top:6px}</style>"
            f"<h2>Pipeline evidence · {datetime.now():%Y-%m-%d %H:%M} · {len(rows)} images</h2>"
            + "".join(cards))
    (out_dir / "summary.html").write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image files and/or folders")
    ap.add_argument("--video", help="take frames from a video instead")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between video frames (default 2)")
    ap.add_argument("--centroid", action="store_true", help="use the centroid decision even if calibrated")
    ap.add_argument("--out", help="output folder (default logs/evidence/<date_time>)")
    ap.add_argument("--lang", choices=["es", "en"], default="es", help="figure labels (default es)")
    args = ap.parse_args()
    global LANG
    LANG = args.lang

    if args.video:
        sources = frames_from_video(args.video, args.every)
    else:
        files = collect_images(args.images)
        if not files:
            sys.exit("No images given. Example: python evidence.py ~/Assistive-Navigation/logs/navigation/snapshots/")
        sources = ((f.stem, cv2.imread(str(f))) for f in files)

    out_dir = Path(args.out) if args.out else C.BASE / "logs" / "evidence" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import bev
    from perception import Perception
    use_bev = not args.centroid and bev.CALIB_FILE.exists()
    if use_bev:
        from planner import BevPlanner as make_decider
    else:
        from decision import Decider as make_decider
    print(f"Loading models... (decision: {'BEV planner' if use_bev else 'centroid'})")
    perception = Perception()

    warm = np.zeros((1080, 1920, 3), np.uint8)       # first inferences are slower (CUDA warm-up)
    for _ in range(3):
        perception.process(warm)

    rows = []
    for name, frame in sources:
        if frame is None:
            print(f"Could not read {name}")
            continue
        if frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080))

        walkable, boxes, times = perception.process(frame)
        mask = perception.last_mask

        # A still image has no history: feed it several times so smoothing/debounce settle,
        # exactly as if the user stood still in front of that scene.
        decider = make_decider()
        t0 = time.perf_counter()
        for _ in range(SETTLE_FRAMES):
            res = (decider.update(walkable, boxes, frame.shape, mask) if use_bev
                   else decider.update(walkable, boxes, frame.shape))
        plan_ms = (time.perf_counter() - t0) * 1000 / SETTLE_FRAMES

        obst = getattr(res, "obst_dist", None)
        row = {"image": name, "command": res.command, "reason": res.reason,
               "angle_deg": round(getattr(res, "angle_deg", 0.0), 1),
               "path_len_m": round(getattr(res, "path_len", 0.0), 2),
               "obstacle_m": None if obst is None else round(obst, 2),
               "sidewalk_pct": round(100 * res.sidewalk_frac, 1),
               "n_boxes": len(boxes), "n_obstacles": len(res.obstacles),
               "seg_ms": round(times["seg_ms"], 1), "det_ms": round(times["det_ms"], 1),
               "plan_ms": round(plan_ms, 2), "figure": f"{name}_result.png"}
        if LANG == "es":
            f1 = lambda v: "-" if v is None else str(v).replace(".", ",")
            lines = [f"{name}   →   {ES_COMMANDS.get(res.command, res.command)}  ({tr_reason(res.reason)})",
                     f"segmentación {f1(row['seg_ms'])} ms | detección {f1(row['det_ms'])} ms | "
                     f"planificador {f1(row['plan_ms'])} ms | acera {f1(row['sidewalk_pct'])} % | "
                     f"objetos {row['n_boxes']} | "
                     + (f"ruta {f1(row['path_len_m'])} m | ángulo {f1(row['angle_deg'])}° | "
                        f"obstáculo {f1(row['obstacle_m'])} m" if use_bev else f"error {res.error:+.2f}")]
        else:
            lines = [f"{name}   ->   {res.command}  ({res.reason})",
                     f"seg {row['seg_ms']} ms | det {row['det_ms']} ms | plan {row['plan_ms']} ms | "
                     f"sidewalk {row['sidewalk_pct']} % | boxes {row['n_boxes']} | "
                     + (f"path {row['path_len_m']} m | angle {row['angle_deg']} deg | obstacle {row['obstacle_m']} m"
                        if use_bev else f"error {res.error:+.2f}")]
        cv2.imwrite(str(out_dir / row["figure"]), compose(frame, walkable, boxes, res, mask, lines))
        rows.append(row)
        print(f"{name:<32s} {res.command}  {res.reason:<28s} seg {row['seg_ms']:5.1f} ms  det {row['det_ms']:5.1f} ms")

    if not rows:
        sys.exit("Nothing processed.")
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_html(out_dir, rows)

    seg = [r["seg_ms"] for r in rows]
    det = [r["det_ms"] for r in rows]
    print(f"\n{len(rows)} images | seg {np.mean(seg):.1f} ± {np.std(seg):.1f} ms | "
          f"det {np.mean(det):.1f} ± {np.std(det):.1f} ms")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
