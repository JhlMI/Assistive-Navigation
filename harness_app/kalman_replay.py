#!/usr/bin/env python3
"""Replay a gps_lab CSV through the Kalman filter and compare raw vs filtered.

    python kalman_replay.py logs/gps/lab/<session>/route_1.csv --refs logs/gps/lab/ref_points.json
    python kalman_replay.py static_1.csv --refs ref_points.json --pos-sigma 3 --accel 0.6

Writes next to the CSV (or in --out): <name>_kf.csv (filtered columns added), <name>_kf.kml
(raw track red, filtered green, references) and <name>_kf.json (metrics). Prints the metrics.

Metrics (raw -> filtered):
  hold / static: dispersion around its own mean (CEP50, p95) and bias vs the reference point.
  walks: roughness = RMS distance to the 5 s centered moving average (high-frequency noise),
         path length / straight length, cross-track mean/RMS vs the true A-B line (bias: the
         filter does NOT remove it), heading noise of compass and GPS course vs the A-B bearing.
"""
import argparse
import csv
import json
import math
import os

import kalman as KF


def num(v):
    return None if v in (None, "", "None") else float(v)


def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("t_s", "lat", "lon", "hdop", "cn0_top4", "speed_kmh", "course_deg", "compass_deg"):
                r[k] = num(r.get(k))
            r["lap"] = int(float(r["lap"])) if r.get("lap") not in (None, "", "None") else None
            rows.append(r)
    return rows


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


def r2(x, nd=2):
    return None if x is None else round(x, nd)


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def disp(pts):
    if len(pts) < 3:
        return None
    me = sum(p[0] for p in pts) / len(pts)
    mn = sum(p[1] for p in pts) / len(pts)
    d = [math.hypot(p[0] - me, p[1] - mn) for p in pts]
    return {"cep50_m": r2(pct(d, 50)), "p95_m": r2(pct(d, 95)), "mean": (me, mn)}


def roughness(pts, t, win=5.0):
    out = []
    j0 = 0
    for i, (p, ti) in enumerate(zip(pts, t)):
        while t[j0] < ti - win / 2:
            j0 += 1
        j1 = i
        while j1 + 1 < len(t) and t[j1 + 1] <= ti + win / 2:
            j1 += 1
        seg = pts[j0:j1 + 1]
        me = sum(q[0] for q in seg) / len(seg)
        mn = sum(q[1] for q in seg) / len(seg)
        out.append(math.hypot(p[0] - me, p[1] - mn))
    return math.sqrt(sum(x * x for x in out) / len(out)) if out else None


def path_len(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def circ_std(d):
    if len(d) < 3:
        return None
    m = sum(d) / len(d)
    return math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))


def kml(path, raw, filt, refs):
    def line(name, color, pts):
        c = " ".join(f"{lon:.7f},{lat:.7f},0" for lat, lon in pts)
        return (f"<Placemark><name>{name}</name><Style><LineStyle><color>{color}</color><width>3</width>"
                f"</LineStyle></Style><LineString><tessellate>1</tessellate><coordinates>{c}</coordinates>"
                f"</LineString></Placemark>")
    pm = "".join(f"<Placemark><name>{k}</name><Point><coordinates>{v['lon']},{v['lat']},0</coordinates></Point></Placemark>"
                 for k, v in (refs or {}).items())
    body = line("GPS crudo", "ff3c3cef", raw) + line("Filtrado (Kalman)", "ff3ccf3c", filt)
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                f'<name>{os.path.basename(path)}</name>{pm}{body}</Document></kml>')


def main():
    ap = argparse.ArgumentParser(description="Replay a gps_lab CSV through the Kalman filter")
    ap.add_argument("csv")
    ap.add_argument("--refs", help="ref_points.json (P, A, B)")
    ap.add_argument("--pos-sigma", type=float, default=KF.POS_SIGMA_M)
    ap.add_argument("--accel", type=float, default=KF.ACCEL_SIGMA)
    ap.add_argument("--out", help="output folder (default: next to the CSV)")
    a = ap.parse_args()

    rows = read(a.csv)
    refs = json.load(open(a.refs)) if a.refs else {}
    kf = KF.GpsKalman(a.pos_sigma, a.accel)
    hf = KF.HeadingFilter()
    for r in rows:
        st = kf.step(r["t_s"], r["lat"], r["lon"], r["hdop"], r["cn0_top4"], r["speed_kmh"], r["course_deg"])
        r["kf"] = st
        r["hdg_f"] = hf.step(r["t_s"], r["compass_deg"])
    rows = [r for r in rows if r["kf"] is not None and r["lat"] is not None]
    if not rows:
        raise SystemExit("no fixes in the CSV")

    loc = kf.to_local
    for r in rows:
        r["raw_en"] = loc(r["lat"], r["lon"])
        r["kf_en"] = (r["kf"]["e"], r["kf"]["n"])
    ref_en = {k: loc(v["lat"], v["lon"]) for k, v in refs.items()}

    m = {"file": os.path.basename(a.csv), "params": {"pos_sigma_m": a.pos_sigma, "accel_sigma": a.accel},
         "fixes": len(rows), "rejected": kf.n_rejected}
    phases = []
    for r in rows:
        key = (r.get("phase"), r.get("lap"))
        if not phases or phases[-1][0] != key:
            phases.append((key, []))
        phases[-1][1].append(r)

    for (phase, lap), seg in phases:
        name = f"{phase}" + (f"_lap{lap}" if lap else "")
        item = {"n": len(seg)}
        if phase in ("record", "signal") or (phase or "").startswith("hold"):
            truth = "P" if phase in ("record", "signal") else ("B" if phase == "hold_B" else "A")
            for tag, key in (("raw", "raw_en"), ("kf", "kf_en")):
                d = disp([r[key] for r in seg])
                if not d:
                    continue
                item[f"{tag}_cep50_m"], item[f"{tag}_p95_m"] = d["cep50_m"], d["p95_m"]
                if truth in ref_en:
                    T = ref_en[truth]
                    item[f"{tag}_bias_m"] = r2(math.hypot(d["mean"][0] - T[0], d["mean"][1] - T[1]))
        elif (phase or "").startswith("walk"):
            t = [r["t_s"] for r in seg]
            for tag, key in (("raw", "raw_en"), ("kf", "kf_en")):
                pts = [r[key] for r in seg]
                item[f"{tag}_roughness_m"] = r2(roughness(pts, t))
                straight = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) or 1.0
                item[f"{tag}_path_ratio"] = r2(path_len(pts) / straight)
            if "A" in ref_en and "B" in ref_en:
                P0, P1 = (ref_en["A"], ref_en["B"]) if phase == "walk_AB" else (ref_en["B"], ref_en["A"])
                dx, dy = P1[0] - P0[0], P1[1] - P0[1]
                L = math.hypot(dx, dy)
                brg = math.degrees(math.atan2(dx, dy)) % 360.0
                for tag, key in (("raw", "raw_en"), ("kf", "kf_en")):
                    xt = [((r[key][0] - P0[0]) * dy - (r[key][1] - P0[1]) * dx) / L for r in seg]
                    item[f"{tag}_xtrack_mean_m"] = r2(sum(xt) / len(xt))
                    item[f"{tag}_xtrack_rms_m"] = r2(math.sqrt(sum(x * x for x in xt) / len(xt)))
                cr = [angle_diff(r["compass_deg"], brg) for r in seg if r["compass_deg"] is not None]
                cf = [angle_diff(r["hdg_f"], brg) for r in seg if r["hdg_f"] is not None]
                gr = [angle_diff(r["course_deg"], brg) for r in seg
                      if r["course_deg"] is not None and (r["speed_kmh"] or 0) >= 2.0]
                gf = [angle_diff(r["kf"]["course_deg"], brg) for r in seg if r["kf"]["course_deg"] is not None]
                item["compass_raw_sigma_deg"], item["compass_f_sigma_deg"] = r2(circ_std(cr), 1), r2(circ_std(cf), 1)
                item["compass_f_bias_deg"] = r2(sum(cf) / len(cf), 1) if cf else None
                item["gps_course_raw_sigma_deg"], item["kf_course_sigma_deg"] = r2(circ_std(gr), 1), r2(circ_std(gf), 1)
        else:
            continue
        m[name] = item

    base = os.path.splitext(os.path.basename(a.csv))[0]
    out = a.out or os.path.dirname(os.path.abspath(a.csv))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, base + "_kf.json"), "w") as f:
        json.dump(m, f, indent=2)
    kml(os.path.join(out, base + "_kf.kml"), [(r["lat"], r["lon"]) for r in rows],
        [(r["kf"]["lat"], r["kf"]["lon"]) for r in rows], refs)
    with open(os.path.join(out, base + "_kf.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "lap", "lat", "lon", "kf_lat", "kf_lon", "kf_speed_kmh", "kf_course_deg",
                    "kf_sigma_m", "still", "compass_deg", "heading_f_deg"])
        for r in rows:
            k = r["kf"]
            w.writerow([r["t_s"], r.get("phase"), r.get("lap"), r["lat"], r["lon"], f"{k['lat']:.7f}",
                        f"{k['lon']:.7f}", f"{k['speed_kmh']:.2f}",
                        "" if k["course_deg"] is None else f"{k['course_deg']:.1f}", f"{k['pos_sigma_m']:.2f}",
                        int(k["still"]), r["compass_deg"], "" if r["hdg_f"] is None else f"{r['hdg_f']:.1f}"])

    print(f"{m['file']}: {m['fixes']} fixes, {m['rejected']} rejected by the gate")
    for k, v in m.items():
        if isinstance(v, dict) and k != "params":
            print(f"  {k:14s} " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
    print(f"Saved {base}_kf.json / .kml / .csv in {out}")


if __name__ == "__main__":
    main()
