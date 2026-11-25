"""Macro-navigation: planning, verified routes and route following (GPS + compass + OSM graph).

Workflow (only verified routes guide a blind user):
  1. plan(dest)            OSM graph (route.py) -> polyline + turn / stairs waypoints
  2. start_verify()        a sighted companion walks it once with the harness; at each planned turn
     mark_corner()         they press "corner" (stores the GPS position there: same bias as later),
     mark_hazard(kind)     and mark what the map does not show (stairs, step, no sidewalk, crossing)
     finish_verify(name)   saved as maps/routes/<name>.json (verified)
  3. start_follow(name)    follower: next waypoint, distance, side, cues, arrival

Cues (shown in the web view now; sent to the chest motors in stage 3):
  TURN_SOON L/R   next turn within CUE_TURN_M
  TURN_NOW  L/R   at the turn (within the arrival radius, or passed it)
  HAZARD          stairs / hazard within CUE_HAZARD_M (combined with a turn when both are close)
  OFF_COURSE L/R  walking and the next waypoint is more than OFF_DEG to one side for OFF_HOLD_S
  OFF_COURSE_BACK walking away from it (more than 135 deg)
  ARRIVED         destination reached
  LOW_GPS         signal too poor to trust the position (no turn cues are produced meanwhile)

The GPS error (3-9 m in the open, 8-12 m between buildings, measured 2026-11-24) sets the radii:
ARRIVE_M 12 m, and cues start well before the point.
"""
import json
import math
import threading
import time
from pathlib import Path

import config as C

ROUTES_DIR = Path(getattr(C, "ROUTES_DIR", C.BASE / "maps" / "routes"))
GRAPH_FILE = Path(getattr(C, "MAP_GRAPH_FILE", C.BASE / "maps" / "lapaz_walk.json"))
ARRIVE_M = getattr(C, "ROUTE_ARRIVE_M", 12.0)
CUE_TURN_M = getattr(C, "ROUTE_CUE_TURN_M", 18.0)   # one turn cue from here until the corner is passed
CUE_HAZARD_M = getattr(C, "ROUTE_CUE_HAZARD_M", 20.0)
COMBINE_M = 15.0                 # a turn and a hazard closer than this give one combined cue
OFF_DEG = 50.0
OFF_HOLD_S = 3.0
GOOD_HDOP = 2.5
GOOD_CN0 = 25.0
HAZARD_KINDS = {"stairs": "escalera", "step": "grada", "no_sidewalk": "sin acera", "crossing": "cruce",
                "obstacle": "obstáculo fijo", "other": "peligro"}
EARTH_K = math.pi / 180.0 * 6371000.0

# Simplified chest interface (2026-11-24): the PLACE says what to do, not the rhythm.
#   left chest  (N,3) = go left   : turn left, or off course with the path to the left
#   right chest (N,4) = go right  : turn right, or off course with the path to the right
#   both sides  (N,5) = hazard ahead (stairs, step, marked hazard)
#   both, once  (N,9) = arrived
# Cue names are kept detailed in the CSV; the user only feels these four messages.
def chest_code(cue):
    if not cue or cue == "LOW_GPS":
        return 0
    if cue == "ARRIVED":
        return 9
    # HAZARD is logged but not sent to the chest: routes are verified to avoid stairs and
    # the vision cannot detect them, so the user only gets direction + arrival.
    if cue.endswith("_L") or "_L+" in cue:
        return 3
    if cue.endswith("_R") or "_R+" in cue:
        return 4
    return 0


def dist_m(a, b):
    return math.hypot((b[1] - a[1]) * EARTH_K * math.cos(math.radians((a[0] + b[0]) / 2)), (b[0] - a[0]) * EARTH_K)


def bearing(a, b):
    e = (b[1] - a[1]) * EARTH_K * math.cos(math.radians(a[0]))
    n = (b[0] - a[0]) * EARTH_K
    return math.degrees(math.atan2(e, n)) % 360.0


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def safe_name(s):
    keep = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in s.strip())
    return keep.replace(" ", "_")[:60] or time.strftime("ruta_%Y%m%d_%H%M%S")


class RouteManager:
    def __init__(self, geonav=None, gps=None):
        self.geonav, self.gps = geonav, gps
        self.lock = threading.Lock()
        self.graph = None
        self.graph_error = None
        self.plan = None           # planned route dict (route.py) + "dest"
        self.verify = None         # verification session
        self.follow = None         # follower state
        self.last_cue = None
        ROUTES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- graph (loaded lazily: 0.4 s) --------------------------------------------
    def _graph(self):
        if self.graph is None and self.graph_error is None:
            try:
                from route import WalkGraph
                self.graph = WalkGraph(str(GRAPH_FILE))
            except Exception as e:
                self.graph_error = f"no se pudo cargar el mapa ({GRAPH_FILE.name}): {e}"
        return self.graph

    def map_segments(self, lat, lon, radius_m=450.0):
        """Walkable ways around a point, for drawing the offline map: [[lat1, lon1, lat2, lon2, kind], ...]."""
        g = self._graph()
        if g is None:
            return {"error": self.graph_error}
        r_deg = radius_m / 111000.0
        ci, cj, rr = int(lat * 1000), int(lon * 1000), int(r_deg * 1000) + 1
        seen, segs = set(), []
        for i in range(ci - rr, ci + rr + 1):
            for j in range(cj - rr, cj + rr + 1):
                for n in g.grid.get((i, j), ()):
                    for m, c, d, kind, name in g.adj[n]:
                        key = (min(n, m), max(n, m))
                        if key in seen:
                            continue
                        seen.add(key)
                        a, b = g.nodes[n], g.nodes[m]
                        if abs(a[0] - lat) < r_deg and abs(a[1] - lon) < r_deg * 1.05:
                            segs.append([round(a[0], 6), round(a[1], 6), round(b[0], 6), round(b[1], 6), kind])
        return {"segments": segs}

    # ---- current position helpers ------------------------------------------------
    def _pos(self):
        return self.geonav.position() if self.geonav else None

    def _quality(self):
        st = {}
        try:
            st = self.gps.status() if self.gps else {}
        except Exception:
            pass
        hdop, cn0 = st.get("gps_hdop"), st.get("gps_cn0_top4")
        good = (hdop is not None and hdop <= GOOD_HDOP) and (cn0 is None or cn0 >= GOOD_CN0)
        return good, hdop, cn0

    # ---- 1. planning ---------------------------------------------------------------
    def plan_route(self, dest, start=None):
        g = self._graph()
        if g is None:
            return {"error": self.graph_error}
        if start is None:
            p = self._pos()
            if p is None:
                return {"error": "sin posición GPS: espera el fix o elige el origen en el mapa"}
            start = (p[0], p[1])
        r = g.route(tuple(start), tuple(dest))
        if "error" in r:
            return r
        r["start"], r["dest"] = list(start), list(dest)
        with self.lock:
            self.plan = r
        return r

    # ---- 2. verification walk ----------------------------------------------------------
    def start_verify(self):
        with self.lock:
            if self.plan is None:
                return {"error": "primero calcula una ruta"}
            turns = [w for w in self.plan["waypoints"] if w["turn"] in ("L", "R")]
            self.verify = {"t0": time.time(), "track": [], "corners": [None] * len(turns), "turns": turns,
                           "next_corner": 0, "hazards": [], "plan": self.plan}
        return {"ok": True, "turns": len(turns)}

    def mark_corner(self):
        with self.lock:
            v = self.verify
            if v is None:
                return {"error": "no hay verificación en curso"}
            p = self._pos()
            if p is None:
                return {"error": "sin posición GPS"}
            if v["next_corner"] >= len(v["turns"]):
                return {"error": "ya se marcaron todos los giros"}
            v["corners"][v["next_corner"]] = [p[0], p[1], round(p[2], 1)]
            v["next_corner"] += 1
            return {"ok": True, "corner": v["next_corner"], "of": len(v["turns"])}

    def mark_hazard(self, kind):
        with self.lock:
            v = self.verify
            if v is None:
                return {"error": "no hay verificación en curso"}
            p = self._pos()
            if p is None:
                return {"error": "sin posición GPS"}
            kind = kind if kind in HAZARD_KINDS else "other"
            v["hazards"].append({"lat": p[0], "lon": p[1], "kind": kind, "t": round(time.time() - v["t0"], 1)})
            return {"ok": True, "hazards": len(v["hazards"])}

    def finish_verify(self, name, arrived=True):
        with self.lock:
            v = self.verify
            if v is None:
                return {"error": "no hay verificación en curso"}
            p = self._pos()
            plan = v["plan"]
            wps = []
            for w, c in zip(v["turns"], v["corners"]):
                ww = dict(w)
                if c:                                   # use where the GPS said the corner was
                    ww.update(lat=c[0], lon=c[1], map_lat=w["lat"], map_lon=w["lon"], marked=True)
                wps.append(ww)
            for w in plan["waypoints"]:                 # stairs from the map
                if w["turn"] == "STEPS":
                    wps.append({**w, "kind": "stairs"})
            for h in v["hazards"]:                      # hazards marked on the walk
                wps.append({"lat": h["lat"], "lon": h["lon"], "turn": "HAZARD", "kind": h["kind"], "marked": True})
            end = {"lat": plan["dest"][0], "lon": plan["dest"][1], "turn": "END"}
            if arrived and p is not None:
                end.update(lat=p[0], lon=p[1], map_lat=plan["dest"][0], map_lon=plan["dest"][1], marked=True)
            # order along the walked track (fall back to the plan order)
            track = v["track"]
            def along(w):
                if len(track) < 2:
                    return w.get("dist_m", 0.0)
                best = min(range(len(track)), key=lambda i: dist_m(track[i][1:3], (w["lat"], w["lon"])))
                return best
            wps.sort(key=along)
            wps.append(end)
            route = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "verified": True,
                     "start": plan["start"], "dest": plan["dest"], "length_m": plan["length_m"],
                     "polyline": plan["polyline"], "walked": [[t[1], t[2]] for t in track],
                     "waypoints": wps, "corners_marked": sum(1 for c in v["corners"] if c),
                     "corners_planned": len(v["turns"]), "hazards_marked": len(v["hazards"])}
            path = ROUTES_DIR / f"{safe_name(name)}.json"
            path.write_text(json.dumps(route, indent=1, ensure_ascii=False))
            self.verify = None
        return {"ok": True, "file": path.name, "waypoints": len(wps)}

    def cancel_verify(self):
        with self.lock:
            self.verify = None
        return {"ok": True}

    # ---- saved routes --------------------------------------------------------------------
    def list_routes(self):
        out = []
        for p in sorted(ROUTES_DIR.glob("*.json")):
            try:
                r = json.loads(p.read_text())
                out.append({"file": p.name, "name": r.get("name", p.stem), "length_m": r.get("length_m"),
                            "created": r.get("created"), "verified": r.get("verified", False),
                            "waypoints": len(r.get("waypoints", []))})
            except Exception:
                continue
        return out

    def load_route(self, file):
        p = ROUTES_DIR / Path(file).name
        return json.loads(p.read_text()) if p.exists() else None

    def delete_route(self, file):
        p = ROUTES_DIR / Path(file).name
        if p.exists():
            p.unlink()
            return {"ok": True}
        return {"error": "no existe"}

    # ---- 3. following -------------------------------------------------------------------------
    def start_follow(self, file):
        r = self.load_route(file)
        if r is None:
            return {"error": "ruta no encontrada"}
        with self.lock:
            self.follow = {"route": r, "file": Path(file).name, "i": 0, "t0": time.time(), "min_d": {},
                           "off_since": None, "events": [], "arrived": False}
            self.last_cue = None
        return {"ok": True, "waypoints": len(r["waypoints"])}

    def stop_follow(self):
        with self.lock:
            self.follow = None
            self.last_cue = None
        return {"ok": True}

    def update(self):
        """Call every frame (cheap). Returns a flat dict for the CSV and the web view."""
        out = {"route": None, "wp_i": None, "wp_type": None, "wp_dist_m": None, "wp_rel_deg": None,
               "cue": None, "cue_text": None}
        with self.lock:
            v = self.verify
            if v is not None:                               # verification: record the walked track
                p = self._pos()
                if p is not None and (not v["track"] or dist_m(v["track"][-1][1:3], p[:2]) > 1.0):
                    v["track"].append((round(time.time() - v["t0"], 1), p[0], p[1]))
                out.update(route="verificando", wp_i=v["next_corner"])
            f = self.follow
            if f is None:
                return out
            r = f["route"]
            wps = r["waypoints"]
            out["route"] = r.get("name")
            if f["arrived"]:
                out.update(cue="ARRIVED", cue_text="llegaste al destino")
                return out
            p = self._pos()
            good, hdop, cn0 = self._quality()
            if p is None or not good:
                out.update(cue="LOW_GPS", cue_text=f"GPS insuficiente (HDOP {hdop}, señal {cn0})")
                self._event(f, out["cue"])
                return out
            heading = self.geonav.heading_deg() if self.geonav else None
            here = (p[0], p[1])
            # advance past waypoints already reached (within the radius, or passed: distance grows again)
            while f["i"] < len(wps):
                w = wps[f["i"]]
                d = dist_m(here, (w["lat"], w["lon"]))
                md = min(f["min_d"].get(f["i"], 1e9), d)
                f["min_d"][f["i"]] = md
                reached = d <= ARRIVE_M
                passed = md <= 2 * ARRIVE_M and d > md + 8.0
                if w["turn"] == "END" and reached:
                    f["arrived"] = True
                    out.update(cue="ARRIVED", cue_text="llegaste al destino", wp_i=f["i"], wp_type="END")
                    self._event(f, "ARRIVED")
                    return out
                if (reached and w["turn"] not in ("L", "R")) or passed:
                    f["i"] += 1
                    continue
                break
            if f["i"] >= len(wps):
                f["arrived"] = True
                return out
            w = wps[f["i"]]
            d = dist_m(here, (w["lat"], w["lon"]))
            brg = bearing(here, (w["lat"], w["lon"]))
            rel = None if heading is None else angle_diff(brg, heading)
            out.update(wp_i=f["i"], wp_type=w["turn"], wp_dist_m=round(d, 1),
                       wp_rel_deg=None if rel is None else round(rel, 0))
            nxt = wps[f["i"] + 1] if f["i"] + 1 < len(wps) else None
            cue, text = None, None
            if w["turn"] in ("L", "R"):
                side = "izquierda" if w["turn"] == "L" else "derecha"
                if d <= ARRIVE_M:
                    cue, text = f"TURN_NOW_{w['turn']}", f"gira a la {side}"
                    if self._close_hazard(w, nxt):
                        cue, text = cue + "+HAZARD", text + " y luego " + self._hazard_name(nxt)
                elif d <= CUE_TURN_M:
                    cue, text = f"TURN_SOON_{w['turn']}", f"giro a la {side} en {d:.0f} m"
            elif w["turn"] in ("STEPS", "HAZARD") and d <= CUE_HAZARD_M:
                cue, text = "HAZARD", f"{self._hazard_name(w)} en {d:.0f} m"
            # off course: walking and the next waypoint is far to one side
            moving = self.geonav and not (self.geonav.status().get("nav_still") or 0)
            if cue is None and rel is not None and moving and abs(rel) > OFF_DEG and d > ARRIVE_M:
                f["off_since"] = f["off_since"] or time.monotonic()
                if time.monotonic() - f["off_since"] >= OFF_HOLD_S:
                    if abs(rel) > 135:      # turn around: towards the side of the smaller turn
                        side = "R" if rel > 0 else "L"
                        cue, text = f"OFF_COURSE_BACK_{side}", f"vas en sentido contrario: date la vuelta ({rel:+.0f}°)"
                    else:
                        side = "R" if rel > 0 else "L"
                        cue, text = (f"OFF_COURSE_{side}",
                                     f"el camino sigue a la {'derecha' if rel > 0 else 'izquierda'} ({rel:+.0f}°)")
            else:
                f["off_since"] = None
            out.update(cue=cue, cue_text=text)
            if cue:
                self._event(f, cue)
            return out

    @staticmethod
    def _close_hazard(w, nxt):
        return (nxt is not None and nxt["turn"] in ("STEPS", "HAZARD")
                and dist_m((w["lat"], w["lon"]), (nxt["lat"], nxt["lon"])) <= COMBINE_M)

    @staticmethod
    def _hazard_name(w):
        if w is None:
            return "peligro"
        if w["turn"] == "STEPS":
            return f"escalera ({w.get('len_m', 0):.0f} m)"
        return HAZARD_KINDS.get(w.get("kind", "other"), "peligro")

    def _event(self, f, cue):
        if cue != self.last_cue:
            f["events"].append((round(time.time() - f["t0"], 1), cue))
            self.last_cue = cue

    # ---- web state ----------------------------------------------------------------------------
    def state(self):
        p = self._pos()
        good, hdop, cn0 = self._quality()
        with self.lock:
            v, f, plan = self.verify, self.follow, self.plan
            st = {"pos": None if p is None else [round(p[0], 7), round(p[1], 7), round(p[2], 1)],
                  "heading": self.geonav.heading_deg() if self.geonav else None,
                  "gps_good": good, "hdop": hdop, "cn0": cn0,
                  "map_ok": self.graph is not None or (self.graph_error is None and GRAPH_FILE.exists()),
                  "map_error": self.graph_error,
                  "plan": None if plan is None else {k: plan[k] for k in ("start", "dest", "length_m", "polyline",
                                                                          "waypoints", "streets", "steps_m")},
                  "verify": None if v is None else {"track": [t[1:] for t in v["track"]][-600:],
                                                    "next_corner": v["next_corner"], "turns": len(v["turns"]),
                                                    "hazards": v["hazards"]},
                  "follow": None if f is None else {"name": f["route"].get("name"), "i": f["i"],
                                                    "waypoints": f["route"]["waypoints"],
                                                    "polyline": f["route"].get("polyline"),
                                                    "arrived": f["arrived"], "events": f["events"][-12:]}}
        return st
