"""Evidence video: replays a raw recording through the full pipeline OFFLINE (every frame,
no real-time pressure) and writes what the system saw and decided, frame by frame:

    top:    camera + sidewalk mask + boxes + IPM path   |  bird's-eye (BEV) panel
    bottom: original frame                              |  segmentation (all classes)
    footer: time in the clip, command, per-stage times, angle, path, obstacle

The planner keeps its temporal state between frames, exactly as in the live system.
Stage times (seg/det/plan) are measured too, so the video doubles as a timing record.

Usage:
    python evidence_video.py ~/Assistive-Navigation/logs/navigation/rec_20261104_155833.avi --fps 22
    python evidence_video.py rec_X.mp4                          # new recordings: real FPS in the file
    python evidence_video.py rec_X.avi --fps 22 --start 30 --end 90 --scale 0.75

    --fps     real FPS of the recording. Recordings made before the re-timed recorder (all .avi
              files up to 2026-11-04) say 15 in the header but were ~22 FPS: pass --fps 22.
    --start / --end   seconds of the clip to process (default: all).
    --scale   output size factor (1.0 = 1760 px wide).

Output: ~/Assistive-Navigation/logs/evidence/<clip>_evidence.mp4 + <clip>_evidence.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

if "ultralytics" in sys.modules:
    sys.exit("ultralytics is imported: it drops OpenCV to 1 thread. Aborting.")

import cv2
import numpy as np

import config as C
from evidence import compose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, help="real FPS of the recording (old .avi files: 22)")
    ap.add_argument("--start", type=float, default=0.0, help="start second")
    ap.add_argument("--end", type=float, help="end second")
    ap.add_argument("--scale", type=float, default=1.0, help="output size factor")
    ap.add_argument("--centroid", action="store_true", help="centroid decision instead of the BEV planner")
    ap.add_argument("--out", help="output folder (default logs/evidence)")
    args = ap.parse_args()

    src = Path(args.video).expanduser()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"Could not open {src}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 15.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(args.start * fps)
    last = min(n_total, int(args.end * fps)) if args.end else n_total
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    out_dir = Path(args.out).expanduser() if args.out else C.BASE / "logs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = src.stem + (f"_{int(args.start)}-{int(args.end)}s" if args.end or args.start else "")
    out_video = out_dir / f"{tag}_evidence.mp4"
    out_csv = out_dir / f"{tag}_evidence.csv"

    import bev
    from perception import Perception
    use_bev = not args.centroid and bev.CALIB_FILE.exists()
    if use_bev:
        from planner import BevPlanner
        decider = BevPlanner()
    else:
        from decision import Decider
        decider = Decider()
    print(f"{src.name}: {n_total} frames at {fps:.1f} FPS -> processing frames {first}..{last} "
          f"({(last - first) / fps:.0f} s) | decision: {'BEV planner' if use_bev else 'centroid'}")
    print("Loading models...")
    perception = Perception()
    warm = np.zeros((1080, 1920, 3), np.uint8)
    for _ in range(3):
        perception.process(warm)

    writer = None
    rows = []
    pipe_ema = None                       # smoothed seg + det + plan time (ms)
    t_start = time.perf_counter()
    for i in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080))

        walkable, boxes, times = perception.process(frame)
        t0 = time.perf_counter()
        res = (decider.update(walkable, boxes, frame.shape, perception.last_mask) if use_bev
               else decider.update(walkable, boxes, frame.shape))
        plan_ms = (time.perf_counter() - t0) * 1000

        pipe_ms = times["seg_ms"] + times["det_ms"] + plan_ms
        pipe_ema = pipe_ms if pipe_ema is None else 0.9 * pipe_ema + 0.1 * pipe_ms
        pipe_fps = 1000.0 / pipe_ema

        t_clip = i / fps
        obst = getattr(res, "obst_dist", None)
        row = {"frame": i, "t_s": round(t_clip, 2), "command": res.command, "reason": res.reason,
               "angle_deg": round(getattr(res, "angle_deg", 0.0), 1),
               "path_len_m": round(getattr(res, "path_len", 0.0), 2),
               "obstacle_m": None if obst is None else round(obst, 2),
               "sidewalk_pct": round(100 * res.sidewalk_frac, 1), "n_boxes": len(boxes),
               "seg_ms": round(times["seg_ms"], 1), "det_ms": round(times["det_ms"], 1),
               "plan_ms": round(plan_ms, 1), "pipeline_ms": round(pipe_ms, 1),
               "pipeline_fps": round(1000.0 / pipe_ms, 1)}
        rows.append(row)

        mm, ss = divmod(t_clip, 60)
        lines = [f"{src.name}  t={int(mm):02d}:{ss:05.2f}   ->   {res.command}  ({res.reason})"
                 f"   |   pipeline {pipe_fps:4.1f} FPS ({pipe_ema:.0f} ms)",
                 f"seg {row['seg_ms']} ms | det {row['det_ms']} ms | plan {row['plan_ms']} ms | "
                 f"sidewalk {row['sidewalk_pct']} % | boxes {row['n_boxes']}"
                 + (f" | path {row['path_len_m']} m | angle {row['angle_deg']} deg | obstacle {row['obstacle_m']} m"
                    if use_bev else f" | error {res.error:+.2f}")]
        img = compose(frame, walkable, boxes, res, perception.last_mask, lines)
        badge = f"{pipe_fps:4.1f} FPS"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x0 = 1280 - tw - 20                                   # right edge of the camera view
        cv2.rectangle(img, (x0 - 8, 36), (x0 + tw + 8, 36 + th + 16), (0, 0, 0), -1)
        cv2.putText(img, badge, (x0, 36 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 80), 2, cv2.LINE_AA)
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        img = img[:h - h % 2, :w - w % 2]                     # even size for the encoder
        if writer is None:
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (img.shape[1], img.shape[0]))
            if not writer.isOpened():
                out_video = out_video.with_suffix(".avi")
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                         (img.shape[1], img.shape[0]))
        writer.write(img)

        done = i - first + 1
        if done % 50 == 0 or i == last - 1:
            el = time.perf_counter() - t_start
            eta = el / done * (last - first - done)
            print(f"  {done}/{last - first} frames | {done / el:4.1f} frames/s | ETA {eta / 60:4.1f} min", flush=True)

    cap.release()
    if writer is not None:
        writer.release()
    if not rows:
        sys.exit("No frames processed.")
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    cmds = {}
    for r in rows:
        cmds[r["command"]] = cmds.get(r["command"], 0) + 1
    for key in ("seg_ms", "det_ms", "plan_ms", "pipeline_ms", "pipeline_fps"):
        v = np.array([r[key] for r in rows])
        unit = "FPS" if key.endswith("fps") else "ms"
        pct = np.percentile(v, 5 if unit == "FPS" else 95)
        print(f"  {key:12s} mean {v.mean():5.1f} {unit}   {'p5 ' if unit == 'FPS' else 'p95'} {pct:5.1f} {unit}")
    print("  commands: " + ", ".join(f"{k} {100 * n / len(rows):.0f} %" for k, n in sorted(cmds.items())))
    print(f"Saved {out_video}\n      {out_csv}")


if __name__ == "__main__":
    main()
