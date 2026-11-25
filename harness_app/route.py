#!/usr/bin/env python3
"""Offline walking routes on the OpenStreetMap graph built by osm_build.py (standard library only).

    python route.py --from=-16.52789,-68.10645 --to=-16.52266,-68.11045
    python route.py --from=... --to=... --kml ruta.kml      # check it on Google Earth
(use --from=... with "=": the coordinates start with "-")

WalkGraph.route() returns the path and the turn waypoints:
  - A* shortest path, with a cost factor per way type (busy avenues and steps cost more).
  - The path is simplified (Douglas-Peucker, 4 m) and every vertex where the direction changes
    by more than TURN_DEG becomes a waypoint with its turn side (L / R) and angle.
  - The last waypoint is the destination.
"""
import argparse
import heapq
import json
import math
import os

EARTH_R = 6371000.0
K = math.pi / 180.0 * EARTH_R
TURN_DEG = 35.0           # direction change that counts as a turn
SIMPLIFY_M = 4.0
MAX_SNAP_M = 80.0         # start/end must be this close to a walkable way
TURN_PENALTY_M = 40.0     # extra cost of changing street (fewer turns for the user)
MERGE_M = 20.0            # turns closer than this are one maneuver (crossings, sidewalk links, jogs)
STEPS_GAP_M = 15.0        # stair flights separated by less than this (landings) are one staircase
COST = {"footway": 1.0, "pedestrian": 1.0, "path": 1.05, "living_street": 1.0, "residential": 1.0,
        "unclassified": 1.0, "service": 1.05, "track": 1.2, "corridor": 1.0, "crossing": 1.0,
        "tertiary": 1.05, "tertiary_link": 1.05, "secondary": 1.1, "secondary_link": 1.1,
        "primary": 1.2, "primary_link": 1.2, "trunk": 1.4, "trunk_link": 1.4, "steps": 2.0}
DEFAULT_GRAPH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "lapaz_walk.json")


def local(lat, lon, lat0, lon0):
    return (lon - lon0) * K * math.cos(math.radians(lat0)), (lat - lat0) * K


def dist_m(a, b):
    return math.hypot((b[1] - a[1]) * K * math.cos(math.radians((a[0] + b[0]) / 2)), (b[0] - a[0]) * K)


def bearing(a, b):
    e, n = local(b[0], b[1], a[0], a[1])
    return math.degrees(math.atan2(e, n)) % 360.0


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


class WalkGraph:
    def __init__(self, path=DEFAULT_GRAPH):
        g = json.load(open(path))
        self.nodes = [tuple(n) for n in g["nodes"]]
        self.names = g["names"]
        self.adj = [[] for _ in self.nodes]
        for a, b, d, kind, name in g["edges"]:
            c = d * COST.get(kind, 1.1)
            self.adj[a].append((b, c, d, kind, name))
            self.adj[b].append((a, c, d, kind, name))
        self.grid = {}
        for i, (lat, lon) in enumerate(self.nodes):
            self.grid.setdefault((int(lat * 1000), int(lon * 1000)), []).append(i)
        self.info = {"nodes": len(self.nodes), "edges": len(g["edges"]), "built": g.get("built"),
                     "source": g.get("source")}

    def nearest(self, lat, lon):
        ci, cj = int(lat * 1000), int(lon * 1000)
        best, bd = None, float("inf")
        for r in range(0, 3):
            for i in range(ci - r, ci + r + 1):
                for j in range(cj - r, cj + r + 1):
                    for n in self.grid.get((i, j), ()):
                        if not self.adj[n]:
                            continue
                        d = dist_m((lat, lon), self.nodes[n])
                        if d < bd:
                            best, bd = n, d
            if best is not None and r >= 1:
                break
        return best, bd

    def _astar(self, s, t):
        """A* over (node, street) states: changing street costs TURN_PENALTY_M, so among routes of
        similar length the one with fewer turns wins (each turn is a decision for the user)."""
        goal = self.nodes[t]
        h = lambda n: dist_m(self.nodes[n], goal)
        start = (s, -1)
        openq = [(h(s), 0.0, start)]
        g = {start: 0.0}
        prev = {}
        end = None
        while openq:
            f, gc, st = heapq.heappop(openq)
            n, cur = st
            if n == t:
                end = st
                break
            if gc > g.get(st, float("inf")):
                continue
            for m, c, d, kind, name in self.adj[n]:
                pen = TURN_PENALTY_M if (cur >= 0 and name != cur and (name or cur)) else 0.0
                ns = (m, name)
                ng = gc + c + pen
                if ng < g.get(ns, float("inf")):
                    g[ns] = ng
                    prev[ns] = (st, d, kind, name)
                    heapq.heappush(openq, (ng + h(m), ng, ns))
        if end is None:
            return (None if s != t else ([s], []))
        path, steps, st = [end[0]], [], end
        while st != start:
            p, d, kind, name = prev[st]
            steps.append((d, kind, name))
            path.append(p[0])
            st = p
        return path[::-1], steps[::-1]

    def route(self, start, goal):
        s, ds = self.nearest(*start)
        t, dt = self.nearest(*goal)
        if s is None or ds > MAX_SNAP_M:
            return {"error": f"el origen está a {ds:.0f} m de la calle más cercana del mapa" if s else "fuera del mapa"}
        if t is None or dt > MAX_SNAP_M:
            return {"error": f"el destino está a {dt:.0f} m de la calle más cercana del mapa" if t else "fuera del mapa"}
        res = self._astar(s, t)
        if res is None:
            return {"error": "no hay camino a pie entre esos puntos en el mapa"}
        path, steps = res
        pts = [self.nodes[n] for n in path]
        street = [self.names[st[2]] for st in steps]      # street of each segment pts[i] -> pts[i+1]
        kinds = [st[1] for st in steps]
        length = sum(st[0] for st in steps)
        keep = simplify(pts, SIMPLIFY_M)
        wps = turns(pts, keep, street)
        # stairs are hazards, not turns: always their own waypoint (never merged away)
        cum = [0.0]
        for st in steps:
            cum.append(cum[-1] + st[0])
        runs, i = [], 0                      # [first segment, last segment, stair metres, flights]
        while i < len(steps):
            if steps[i][1] == "steps":
                j = i
                while j + 1 < len(steps) and steps[j + 1][1] == "steps":
                    j += 1
                m = cum[j + 1] - cum[i]
                if runs and cum[i] - cum[runs[-1][1] + 1] < STEPS_GAP_M:   # landing between flights
                    runs[-1][1], runs[-1][2], runs[-1][3] = j, runs[-1][2] + m, runs[-1][3] + 1
                else:
                    runs.append([i, j, m, 1])
                i = j + 1
            else:
                i += 1
        for a0, a1, m, n in runs:
            wps.append({"lat": pts[a0][0], "lon": pts[a0][1], "turn": "STEPS", "angle": 0.0,
                        "len_m": round(m, 1), "span_m": round(cum[a1 + 1] - cum[a0], 1), "flights": n,
                        "street": street[a0], "next_street": "", "dist_m": round(cum[a0], 1)})
        wps.sort(key=lambda w: w["dist_m"])
        wps.append({"lat": goal[0], "lon": goal[1], "turn": "END", "angle": 0.0,
                    "street": "", "next_street": "", "dist_m": round(length + dt, 1)})
        return {"length_m": round(length + ds + dt, 1), "snap_start_m": round(ds, 1), "snap_end_m": round(dt, 1),
                "polyline": [list(p) for p in pts], "waypoints": wps,
                "steps_m": round(sum(st[0] for st in steps if st[1] == "steps"), 1),
                "streets": [s for i, s in enumerate(street) if s and (i == 0 or street[i - 1] != s)],
                "main_road_m": round(sum(st[0] for st in steps if st[1] in ("primary", "trunk", "secondary")), 1),
                "kinds": sorted(set(kinds))}


def simplify(pts, tol):
    """Douglas-Peucker on the polyline; returns the indices kept."""
    if len(pts) < 3:
        return list(range(len(pts)))
    lat0, lon0 = pts[0]
    xy = [local(p[0], p[1], lat0, lon0) for p in pts]
    keep = {0, len(pts) - 1}
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = xy[i], xy[j]
        L = math.hypot(x2 - x1, y2 - y1) or 1e-9
        best, bk = -1.0, None
        for k in range(i + 1, j):
            x0, y0 = xy[k]
            d = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / L
            if d > best:
                best, bk = d, k
        if bk is not None and best > tol:
            keep.add(bk)
            stack += [(i, bk), (bk, j)]
    return sorted(keep)


def turns(pts, keep, street):
    out, cum = [], [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + dist_m(a, b))
    for a, k, b in zip(keep, keep[1:], keep[2:]):
        h_in, h_out = bearing(pts[a], pts[k]), bearing(pts[k], pts[b])
        ang = angle_diff(h_out, h_in)
        if abs(ang) >= TURN_DEG:
            out.append({"lat": pts[k][0], "lon": pts[k][1], "turn": "R" if ang > 0 else "L",
                        "angle": round(ang, 1), "heading_in": round(h_in, 1), "heading_out": round(h_out, 1),
                        "street": street[k - 1] if k - 1 < len(street) else "",
                        "next_street": street[k] if k < len(street) else "", "dist_m": round(cum[k], 1)})
    return merge_turns(out)


def merge_turns(raw, merge_m=MERGE_M):
    """Group turns closer than merge_m into one maneuver with the net direction change.
    A group whose net change is small (left-right jog: crossing a street, stepping onto a sidewalk)
    is not a turn and is dropped."""
    out, i = [], 0
    while i < len(raw):
        j = i
        while j + 1 < len(raw) and raw[j + 1]["dist_m"] - raw[j]["dist_m"] < merge_m:
            j += 1
        grp = raw[i:j + 1]
        net = angle_diff(grp[-1]["heading_out"], grp[0]["heading_in"])
        if abs(net) >= TURN_DEG:
            mid = grp[len(grp) // 2]
            out.append({**mid, "turn": "R" if net > 0 else "L", "angle": round(net, 1),
                        "heading_in": grp[0]["heading_in"], "heading_out": grp[-1]["heading_out"],
                        "street": grp[0]["street"], "next_street": grp[-1]["next_street"],
                        "merged": len(grp)})
        i = j + 1
    return out


def to_kml(route, path, name="Ruta"):
    line = " ".join(f"{p[1]:.7f},{p[0]:.7f},0" for p in route["polyline"])
    pms = "".join(
        f"<Placemark><name>{i + 1}: {w['turn']} "
        f"{(str(round(w['len_m'])) + ' m') if w['turn'] == 'STEPS' else format(w.get('angle', 0), '+.0f') + '°'}"
        f" ({w['dist_m']:.0f} m)</name>"
        f"<description>{w.get('street', '')} → {w.get('next_street', '')}</description>"
        f"<Point><coordinates>{w['lon']},{w['lat']},0</coordinates></Point></Placemark>"
        for i, w in enumerate(route["waypoints"]))
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                f'<name>{name}</name><Placemark><name>{name} ({route["length_m"]:.0f} m)</name><Style><LineStyle>'
                f'<color>ff00a5ff</color><width>4</width></LineStyle></Style><LineString><tessellate>1</tessellate>'
                f'<coordinates>{line}</coordinates></LineString></Placemark>{pms}</Document></kml>')


def parse_ll(s):
    a, b = [float(x) for x in s.replace(" ", "").split(",")[:2]]
    return a, b


def main():
    ap = argparse.ArgumentParser(description="Offline walking route")
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--from", dest="src", required=True, help="lat,lon")
    ap.add_argument("--to", dest="dst", required=True, help="lat,lon")
    ap.add_argument("--kml", help="write the route as KML")
    a = ap.parse_args()
    import time
    t0 = time.time()
    g = WalkGraph(a.graph)
    t1 = time.time()
    r = g.route(parse_ll(a.src), parse_ll(a.dst))
    t2 = time.time()
    print(f"Graph: {g.info['nodes']} nodes, {g.info['edges']} edges (load {t1 - t0:.1f} s); route {1000 * (t2 - t1):.0f} ms")
    if "error" in r:
        raise SystemExit(r["error"])
    print(f"Length {r['length_m']:.0f} m | snap start {r['snap_start_m']} m, end {r['snap_end_m']} m | "
          f"main roads {r['main_road_m']:.0f} m | steps {r['steps_m']:.0f} m")
    print("Streets: " + " → ".join(r["streets"]))
    for i, w in enumerate(r["waypoints"], 1):
        if w["turn"] == "STEPS":
            print(f"  {i:2d}. at {w['dist_m']:6.0f} m  STEPS  {w['len_m']:.0f} m of stairs in {w['flights']} flight(s) "
                  f"over {w['span_m']:.0f} m  ({w['lat']:.6f}, {w['lon']:.6f})")
            continue
        print(f"  {i:2d}. at {w['dist_m']:6.0f} m  {w['turn']:3s} {w.get('angle', 0):+6.0f}°  "
              f"{w.get('street', '')} → {w.get('next_street', '')}  ({w['lat']:.6f}, {w['lon']:.6f})")
    if a.kml:
        to_kml(r, a.kml)
        print(f"Saved {a.kml}")


if __name__ == "__main__":
    main()
