"""Breakdown of the perception time per sub-step (pre, h2d, gpu, d2h, post) on real frames.
Usage: python profile_perception.py <video> [--frames 300]   (stop the app first)"""
import argparse
import time

import cv2
import numpy as np

import config as C
from perception import Perception, letterbox, postprocess_detection, walkable_sidewalk


def timed_infer(engine, x):
    torch = engine.torch
    t0 = time.perf_counter()
    with torch.cuda.stream(engine.stream):
        engine.buffers[engine.input].copy_(torch.from_numpy(x))
    engine.stream.synchronize()
    t1 = time.perf_counter()
    with torch.cuda.stream(engine.stream):
        engine.context.execute_async_v3(engine.stream.cuda_stream)
    engine.stream.synchronize()
    t2 = time.perf_counter()
    out = engine.buffers[engine.output].cpu().numpy()
    t3 = time.perf_counter()
    return out, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--skip", type=int, default=30)
    args = ap.parse_args()
    p = Perception()
    cap = cv2.VideoCapture(args.video)
    keys = ["pre", "h2d", "gpu", "d2h", "post", "total"]
    rows = {f"{m}_{k}": [] for m in ("seg", "det") for k in keys}
    n = 0
    while n < args.frames + args.skip:
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if img.shape[1] != 1920:
            img = cv2.resize(img, (1920, 1080))
        t0 = time.perf_counter()
        x, _, l, tp, nh, nw = letterbox(img, C.SEG_H, C.SEG_W)
        t1 = time.perf_counter()
        out, h2d, gpu, d2h = timed_infer(p.seg, x)
        t2 = time.perf_counter()
        mask = np.asarray(out).reshape(C.SEG_H, C.SEG_W).astype(np.uint8)[tp:tp + nh, l:l + nw]
        walkable_sidewalk(mask)
        t3 = time.perf_counter()
        seg = [(t1 - t0) * 1e3, h2d, gpu, d2h, (t3 - t2) * 1e3, (t3 - t0) * 1e3]
        t0 = time.perf_counter()
        xd, rd, ld, td, _, _ = letterbox(img, C.DET_H, C.DET_W)
        t1 = time.perf_counter()
        out, h2d, gpu, d2h = timed_infer(p.det, xd)
        t2 = time.perf_counter()
        postprocess_detection(out, rd, ld, td, img.shape[:2])
        t3 = time.perf_counter()
        det = [(t1 - t0) * 1e3, h2d, gpu, d2h, (t3 - t2) * 1e3, (t3 - t0) * 1e3]
        n += 1
        if n > args.skip:
            for k, v in zip(keys, seg):
                rows["seg_" + k].append(v)
            for k, v in zip(keys, det):
                rows["det_" + k].append(v)
    print(f"\n{args.frames} frames ({args.video})\n")
    print(f"{'step':8s} {'seg mean':>10s} {'seg p95':>10s} {'det mean':>10s} {'det p95':>10s}")
    for k in keys:
        s, d = np.array(rows["seg_" + k]), np.array(rows["det_" + k])
        print(f"{k:8s} {s.mean():8.2f}ms {np.percentile(s, 95):8.2f}ms {d.mean():8.2f}ms {np.percentile(d, 95):8.2f}ms")


if __name__ == "__main__":
    main()
