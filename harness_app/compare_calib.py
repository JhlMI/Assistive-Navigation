"""Compare several IPM calibrations and build a final, averaged one.

Every "Calibrate" press is logged in bev_calibration_runs.jsonl. This script:
  1. Shows how much the runs disagree (in meters) at 2, 3, 4 and 5 m ahead.
  2. Averages the clicked corners of the selected runs (same rectangle and same camera
     position) and saves the result as bev_calibration.json.

Usage:
    python compare_calib.py                 # all runs
    python compare_calib.py --last 5        # only the last 5 runs
    python compare_calib.py --list          # list the runs
    python compare_calib.py --runs 0 2 3    # pick runs by index
    python compare_calib.py --clear         # start a new comparison (deletes the run log)
"""
import argparse

import numpy as np

import bev

TEST_POINTS = [(x, y) for y in (2.0, 3.0, 4.0, 5.0) for x in (-1.0, 0.0, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int)
    ap.add_argument("--runs", type=int, nargs="+")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="only compare, do not overwrite the calibration")
    args = ap.parse_args()

    if args.clear:
        bev.RUNS_FILE.unlink(missing_ok=True)
        print("Run log cleared.")
        return

    runs = bev.load_runs()
    if not runs:
        print("No calibration runs yet. Calibrate at least 2 photos first.")
        return
    indexed = list(enumerate(runs))
    if args.list:
        for i, r in indexed:
            print(f"{i:3d}  {r['time']}  rect={r['rect_m']}  {r.get('photo')}")
        return
    if args.runs:
        indexed = [(i, r) for i, r in indexed if i in args.runs]
    elif args.last:
        indexed = indexed[-args.last:]

    # Only runs with the same rectangle can be averaged
    rects = [tuple(r["rect_m"]) for _, r in indexed]
    rect = max(set(rects), key=rects.count)
    skipped = [i for i, r in indexed if tuple(r["rect_m"]) != rect]
    indexed = [(i, r) for i, r in indexed if tuple(r["rect_m"]) == rect]
    if skipped:
        print(f"Skipping runs {skipped}: different rectangle than {rect}")
    if len(indexed) < 2:
        print("Need at least 2 runs with the same rectangle to compare.")
        return

    pts = np.array([r["points_px"] for _, r in indexed], dtype=np.float64)   # (n, 4, 2)
    mean_pts = pts.mean(axis=0)
    H_mean = bev.homography_from_points(mean_pts, *rect)
    H_inv = np.linalg.inv(H_mean)

    # Pixel positions of the test ground points under the averaged calibration
    test_px = []
    for gx, gy in TEST_POINTS:
        p = H_inv @ np.array([gx, gy, 1.0])
        test_px.append((p[0] / p[2], p[1] / p[2]))

    print(f"\n{len(indexed)} runs, rectangle width={rect[0]} length={rect[1]} near={rect[2]} m")
    print(f"Click spread (std of the corners): {pts.std(axis=0).mean():.1f} px\n")
    print("Disagreement with the average calibration (meters):")
    print(f"{'run':>4s} {'photo':40s} " + " ".join(f"{'@'+str(int(y))+'m':>7s}" for y in (2, 3, 4, 5)))
    worst = {y: 0.0 for y in (2.0, 3.0, 4.0, 5.0)}
    for i, r in indexed:
        H = bev.homography_from_points(r["points_px"], *rect)
        errs = {}
        for (gx, gy), (u, v) in zip(TEST_POINTS, test_px):
            x, y = bev.image_to_ground(H, u, v)
            e = float(np.hypot(x - gx, y - gy))
            errs[gy] = max(errs.get(gy, 0.0), e)
            worst[gy] = max(worst[gy], e)
        photo = (r.get("photo") or "-")[:40]
        print(f"{i:4d} {photo:40s} " + " ".join(f"{errs[y]:7.3f}" for y in (2.0, 3.0, 4.0, 5.0)))
    print("\nWorst case: " + ", ".join(f"{100 * worst[y]:.1f} cm at {y:.0f} m" for y in worst))
    if worst[5.0] > 0.25:
        print("Large spread: the camera moved between photos, or some clicks are off. "
              "Check with --list and pick runs with --runs.")

    if not args.no_save:
        bev.save_calibration(H_mean, {"points_px": mean_pts.tolist(), "rect_m": list(rect),
                                      "averaged_runs": [i for i, _ in indexed]})
        _, y_bottom = bev.image_to_ground(H_mean, 960, 1079)
        print(f"\nAveraged calibration saved to {bev.CALIB_FILE}")
        print(f"Bottom edge of the image (blind zone): {y_bottom:.2f} m ahead of the feet")


if __name__ == "__main__":
    main()
