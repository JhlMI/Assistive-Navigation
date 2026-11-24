#!/usr/bin/env python3
"""Magnetic interference of the vibration motors on the compass.

Drives the motors through the Pico (S, L, R, X, S) while reading the HMC5883L,
logs every sample with its phase and prints the heading / |B| shift per phase.

Run with harness_app stopped (the Pico serial port must be free):
  python motor_compass_test.py --flip --mount 90
"""
import argparse
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pico import Pico  # noqa: E402

PHASES = [("S", "off"), ("L", "left"), ("R", "right"), ("X", "both"), ("S", "off_end")]


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def circ_mean(angles):
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c)) % 360.0


def main():
    ap = argparse.ArgumentParser(description="Motor vs compass interference test")
    ap.add_argument("--scripts-dir", default=os.path.join(HERE, "..", "scripts"),
                    help="folder with compass_probe.py and compass_cal.json")
    ap.add_argument("--phase-s", type=float, default=20.0, help="seconds per phase")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--mount", type=float, default=0.0)
    ap.add_argument("--declination", type=float, default=0.0)
    ap.add_argument("--log", default="motores.csv")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.scripts_dir))
    import compass_probe as cp
    cp.CAL_FILE = os.path.join(os.path.abspath(a.scripts_dir), "compass_cal.json")
    cal = cp.load_cal()
    if not cal:
        sys.exit(f"No calibration at {cp.CAL_FILE}: run compass_probe.py --calibrate 60 first")

    from smbus2 import SMBus
    pico = Pico()
    print(f"Pico on {pico.port}. Stand still, facing the same direction.")
    rows = []
    t0 = time.time()
    try:
        with SMBus(cp.BUS) as bus:
            sensor = cp.detect(bus)
            for letter, name in PHASES:
                print(f"\n== {name} ({letter}) for {a.phase_s:.0f} s ==")
                t_end = time.time() + a.phase_s
                while time.time() < t_end:
                    pico.command(letter)  # keep it fresh (stale watchdog = 0.5 s)
                    v = sensor.read()
                    if v:
                        c = cp.apply_cal(v, cal)
                        h = cp.heading_deg(c[0], c[1], a.declination, a.flip, a.mount)
                        norm = math.sqrt(sum(i * i for i in v))
                        rows.append((time.time() - t0, name, v[0], v[1], v[2], norm, h))
                        print(f"\r  heading {h:6.1f} deg  |B| {norm:5.1f} uT", end="", flush=True)
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pico.command("S")
        time.sleep(0.3)
        pico.close()

    with open(a.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "x_uT", "y_uT", "z_uT", "norm_uT", "heading_deg"])
        for r in rows:
            w.writerow([f"{r[0]:.2f}", r[1]] + [f"{x:.2f}" for x in r[2:6]] + [f"{r[6]:.1f}"])
    print(f"\nSaved {a.log}")

    base = [r for r in rows if r[1] == "off"][3:]  # skip the first samples
    if not base:
        return
    h0 = circ_mean([r[6] for r in base])
    b0 = sum(r[5] for r in base) / len(base)
    print(f"\n{'phase':8s} {'n':>4s} {'heading':>8s} {'shift':>7s} {'max':>6s} {'|B|':>6s} {'d|B|':>6s}")
    for _, name in PHASES:
        ph = [r for r in rows if r[1] == name][3:]
        if not ph:
            continue
        hm = circ_mean([r[6] for r in ph])
        bm = sum(r[5] for r in ph) / len(ph)
        mx = max(abs(angle_diff(r[6], h0)) for r in ph)
        print(f"{name:8s} {len(ph):4d} {hm:8.1f} {angle_diff(hm, h0):+7.1f} {mx:6.1f} {bm:6.1f} {bm - b0:+6.1f}")


if __name__ == "__main__":
    main()
