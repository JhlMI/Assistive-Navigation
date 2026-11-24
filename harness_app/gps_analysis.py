"""Analysis of the GPS field tests recorded by gps_lab.py (also usable offline on its CSV files).

All positions are converted to meters (east, north) around a reference point with an
equirectangular projection (error < 1 cm over a few km).

    python gps_analysis.py logs/gps/lab/<session>/static_1.csv --ref logs/gps/lab/ref_points.json
"""
import argparse
import csv
import json
import math

EARTH_R = 6371000.0


# ---- helpers ---------------------------------------------------------------------
def to_local(lat, lon, lat0, lon0):
    k = math.pi / 180.0 * EARTH_R
    return (lon - lon0) * k * math.cos(math.radians(lat0)), (lat - lat0) * k


def pct(values, p):
    v = sorted(values)
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))
    return v[i]


def mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def std(values):
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def r2(x, nd=2):
    return None if x is None else round(x, nd)


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, 90 = east) of the vector (east, north)."""
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def scatter(pts):
    """Precision of a cloud of (e, n) points around its own mean."""
    if not pts:
        return {}
    me, mn = mean([p[0] for p in pts]), mean([p[1] for p in pts])
    d = [math.hypot(e - me, n - mn) for e, n in pts]
    return {"n": len(pts), "mean_e": me, "mean_n": mn,
            "cep50_m": r2(pct(d, 50)), "p95_m": r2(pct(d, 95)), "max_m": r2(max(d)),
            "rms_m": r2(math.sqrt(sum(x * x for x in d) / len(d))),
            "sigma_e_m": r2(std([p[0] for p in pts])), "sigma_n_m": r2(std([p[1] for p in pts]))}


def quality(rows):
    return {"cn0_top4_mean": r2(mean([r.get("cn0_top4") for r in rows]), 1),
            "cn0_top4_min": r2(min([r["cn0_top4"] for r in rows if r.get("cn0_top4") is not None], default=None), 1),
            "sats_mean": r2(mean([r.get("sats") for r in rows]), 1),
            "sats_view_mean": r2(mean([r.get("sats_view") for r in rows]), 1),
            "hdop_mean": r2(mean([r.get("hdop") for r in rows])),
            "hdop_max": r2(max([r["hdop"] for r in rows if r.get("hdop") is not None], default=None))}


def origin(refs, *names, rows=None):
    for n in names:
        if refs.get(n):
            return refs[n]["lat"], refs[n]["lon"]
    fixed = [r for r in (rows or []) if r.get("lat") is not None]
    return (fixed[0]["lat"], fixed[0]["lon"]) if fixed else (0.0, 0.0)


def local_ref(refs, name, o):
    if not refs.get(name):
        return None
    return to_local(refs[name]["lat"], refs[name]["lon"], *o)


# ---- tests -----------------------------------------------------------------------
def analyze_signal(rows):
    return {"test": "signal", "duration_s": r2(rows[-1]["t_s"] - rows[0]["t_s"], 0) if rows else 0,
            "fixes": len([r for r in rows if r.get("lat") is not None]), **quality(rows)}


def analyze_static(rows, refs):
    rec = [r for r in rows if r.get("phase") == "record" and r.get("lat") is not None]
    if len(rec) < 10:
        return {"test": "static", "error": "fewer than 10 fixes in the recording phase"}
    o = origin(refs, "P", rows=rec)
    pts = [to_local(r["lat"], r["lon"], *o) for r in rec]
    sc = scatter(pts)
    out = {"test": "static", "duration_s": r2(rec[-1]["t_s"] - rec[0]["t_s"], 0), **quality(rec),
           "precision": {k: v for k, v in sc.items() if not k.startswith("mean_")}}
    truth = local_ref(refs, "P", o)
    if truth:
        bias_e, bias_n = sc["mean_e"] - truth[0], sc["mean_n"] - truth[1]
        err = [math.hypot(e - truth[0], n - truth[1]) for e, n in pts]
        out["accuracy_vs_P"] = {"bias_m": r2(math.hypot(bias_e, bias_n)), "bias_e_m": r2(bias_e),
                                "bias_n_m": r2(bias_n), "bias_dir_deg": r2(bearing_deg(bias_e, bias_n), 0),
                                "cep50_m": r2(pct(err, 50)), "p95_m": r2(pct(err, 95)), "max_m": r2(max(err))}
    spd = [r["speed_kmh"] for r in rec if r.get("speed_kmh") is not None]
    if spd:
        out["false_speed_kmh"] = {"mean": r2(mean(spd)), "p95": r2(pct(spd, 95)), "max": r2(max(spd))}
    # Convergence: 5 s window means; settled after the last window farther than 1.5 m from the final mean
    allfix = [r for r in rows if r.get("lat") is not None]
    if allfix:
        t0, last_bad, k = allfix[0]["t_s"], None, 0
        while k < len(allfix):
            seg = [x for x in allfix[k:] if x["t_s"] < allfix[k]["t_s"] + 5.0]
            p = [to_local(x["lat"], x["lon"], *o) for x in seg]
            if math.hypot(mean([q[0] for q in p]) - sc["mean_e"], mean([q[1] for q in p]) - sc["mean_n"]) > 1.5:
                last_bad = seg[-1]["t_s"] - t0
            k += len(seg)
        out["settle_s"] = r2(last_bad or 0.0, 0)
    return out


def _hold_mark(rows, phase, o):
    pts = [to_local(r["lat"], r["lon"], *o) for r in rows if r.get("phase") == phase and r.get("lat") is not None]
    return scatter(pts) if len(pts) >= 3 else None


def analyze_lap(rows, refs, lap):
    lr = [r for r in rows if r.get("lap") == lap]
    if not lr:
        return {"lap": lap, "error": "no data"}
    o = origin(refs, "A", rows=lr)
    A, B = local_ref(refs, "A", o), local_ref(refs, "B", o)
    out = {"lap": lap, **quality([r for r in lr if r.get("lat") is not None])}
    marks = {}
    for name, phase in (("A_start", "hold_A"), ("B", "hold_B"), ("A_end", "hold_A2")):
        m = _hold_mark(lr, phase, o)
        if m:
            marks[name] = m
            truth = A if name.startswith("A") else B
            item = {"hold_cep50_m": m["cep50_m"], "hold_p95_m": m["p95_m"], "n": m["n"]}
            if truth:
                item["error_vs_truth_m"] = r2(math.hypot(m["mean_e"] - truth[0], m["mean_n"] - truth[1]))
            out[f"mark_{name}"] = item
    if "A_start" in marks and "A_end" in marks:
        a1, a2 = marks["A_start"], marks["A_end"]
        out["repeatability_A_m"] = r2(math.hypot(a1["mean_e"] - a2["mean_e"], a1["mean_n"] - a2["mean_n"]))
    if "A_start" in marks and "B" in marks:
        d_meas = math.hypot(marks["B"]["mean_e"] - marks["A_start"]["mean_e"],
                            marks["B"]["mean_n"] - marks["A_start"]["mean_n"])
        out["AB_measured_m"] = r2(d_meas)
        if A and B:
            d_true = math.hypot(B[0] - A[0], B[1] - A[1])
            out["AB_true_m"] = r2(d_true)
            out["AB_error_m"] = r2(d_meas - d_true)
    # Walking segments: cross-track error and heading noise
    for seg, p_from, p_to in (("walk_AB", "A", "B"), ("walk_BA", "B", "A")):
        w = [r for r in lr if r.get("phase") == seg and r.get("lat") is not None]
        if len(w) < 5:
            continue
        P0 = A if p_from == "A" else B
        P1 = B if p_to == "B" else A
        if not (P0 and P1):                         # no truth: use the measured marks
            mk0 = marks.get("A_start" if p_from == "A" else "B")
            mk1 = marks.get("B" if p_to == "B" else "A_end")
            if not (mk0 and mk1):
                continue
            P0, P1 = (mk0["mean_e"], mk0["mean_n"]), (mk1["mean_e"], mk1["mean_n"])
        dx, dy = P1[0] - P0[0], P1[1] - P0[1]
        L = math.hypot(dx, dy) or 1.0
        xt = []
        for r in w:
            e, n = to_local(r["lat"], r["lon"], *o)
            xt.append(((e - P0[0]) * dy - (n - P0[1]) * dx) / L)      # + = right of the walking direction
        true_brg = bearing_deg(dx, dy)
        crs = [angle_diff(r["course_deg"], true_brg) for r in w
               if r.get("course_deg") is not None and (r.get("speed_kmh") or 0) >= 2.0]
        spd = [r["speed_kmh"] for r in w if r.get("speed_kmh") is not None]
        out[seg] = {"duration_s": r2(w[-1]["t_s"] - w[0]["t_s"], 0), "speed_kmh_mean": r2(mean(spd)),
                    "cross_track_rms_m": r2(math.sqrt(sum(x * x for x in xt) / len(xt))),
                    "cross_track_mean_m": r2(mean(xt)), "cross_track_max_m": r2(max(abs(x) for x in xt)),
                    "heading_bias_deg": r2(mean(crs), 1), "heading_sigma_deg": r2(std(crs), 1),
                    "heading_samples": len(crs)}
    return out


def suggestions(static, laps):
    """Filter parameters derived from the measurements (to be checked, not final)."""
    s = {}
    prec = (static or {}).get("precision", {})
    if prec.get("sigma_e_m") is not None:
        s["meas_sigma_m"] = r2(max(prec["sigma_e_m"], prec["sigma_n_m"]))
    fs = (static or {}).get("false_speed_kmh")
    if fs:
        s["still_speed_kmh"] = r2(max(1.0, 1.2 * fs["p95"]), 1)
    if (static or {}).get("settle_s") is not None:
        s["settle_s"] = static["settle_s"]
    errs = [l[k]["error_vs_truth_m"] for l in laps for k in ("mark_A_start", "mark_B", "mark_A_end")
            if isinstance(l.get(k), dict) and l[k].get("error_vs_truth_m") is not None]
    if errs:
        s["arrival_radius_m"] = r2(max(5.0, 1.5 * max(errs)), 1)
    hs = [l[k]["heading_sigma_deg"] for l in laps for k in ("walk_AB", "walk_BA")
          if isinstance(l.get(k), dict) and l[k].get("heading_sigma_deg") is not None]
    if hs:
        s["heading_sigma_deg"] = r2(mean(hs), 1)
    return s


# ---- CSV ---------------------------------------------------------------------------
NUM = ("t_s", "lat", "lon", "fix", "sats", "hdop", "alt_m", "speed_kmh", "course_deg", "cn0_top4", "sats_view", "lap")


def read_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in NUM:
                v = r.get(k)
                r[k] = None if v in (None, "", "None") else (int(float(v)) if k in ("fix", "sats", "sats_view", "lap")
                                                            else float(v))
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ref", help="ref_points.json with P, A, B")
    args = ap.parse_args()
    rows = read_rows(args.csv)
    refs = json.load(open(args.ref)) if args.ref else {}
    test = rows[0].get("test") if rows else ""
    if test == "static":
        res = analyze_static(rows, refs)
    elif test == "route":
        res = [analyze_lap(rows, refs, lap) for lap in sorted({r["lap"] for r in rows if r.get("lap")})]
    else:
        res = analyze_signal(rows)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
