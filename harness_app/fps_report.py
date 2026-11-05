"""FPS diagnosis from a navigation CSV: where the time goes, in windows of N seconds.

Usage:
    python fps_report.py                      # latest nav_*.csv
    python fps_report.py logs/navigation/nav_X.csv --window 10

How to read it:
    lat_ms  = processing time of one frame in the main loop (inference + planning).
    fps     = frames actually processed per second.
    If fps drops but lat_ms stays ~42 ms  -> the loop is WAITING for frames: the camera delivers
                                             fewer (check cam_fps / exp_ms: low light), or USB.
    If lat_ms grows (seg/det grow)        -> the GPU/CPU is slower: thermal or power limits (tj_c),
                                             or other work competing (rec, web_clients).
"""
import argparse
import csv
import sys
from pathlib import Path

import config as C

COLS = ["fps", "lat_ms", "seg_ms", "det_ms", "plan_ms", "cam_fps", "exp_ms", "iso", "tj_c", "v_ina", "rec", "web_clients"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="nav_*.csv (default: the latest one)")
    ap.add_argument("--window", type=float, default=10.0, help="seconds per row (default 10)")
    args = ap.parse_args()

    path = Path(args.csv) if args.csv else max(C.LOG_DIR.glob("nav_*.csv"), key=lambda p: p.stat().st_mtime, default=None)
    if path is None or not path.exists():
        sys.exit("No CSV found")
    rows = list(csv.DictReader(open(path)))
    rows = [r for r in rows if r.get("reason") != "guidance off"]       # only frames with inference
    if not rows:
        sys.exit("No frames with guidance ON in this CSV")
    present = [c for c in COLS if c in rows[0]]
    print(f"{path.name}: {len(rows)} frames with guidance ON\n")
    print(f"{'t (s)':>9} " + " ".join(f"{c:>8}" for c in present))

    t0, bucket = None, []

    def flush(t_start):
        vals = [mean(num(r.get(c)) for r in bucket) for c in present]
        print(f"{t_start:9.0f} " + " ".join("       -" if v is None else f"{v:8.1f}" for v in vals))

    for r in rows:
        t = num(r["t_s"]) or 0.0
        if t0 is None:
            t0 = t
        if t - t0 >= args.window:
            flush(t0)
            t0, bucket = t, []
        bucket.append(r)
    if bucket:
        flush(t0)

    print("\nWhole run:")
    for c in present:
        v = [num(r.get(c)) for r in rows]
        v = [x for x in v if x is not None]
        if v:
            v.sort()
            print(f"  {c:12s} mean {sum(v) / len(v):7.1f}   min {v[0]:7.1f}   p50 {v[len(v) // 2]:7.1f}   max {v[-1]:7.1f}")


if __name__ == "__main__":
    main()
