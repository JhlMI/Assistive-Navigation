#!/usr/bin/env python3
"""Download the walkable street network of an area from OpenStreetMap ONCE and save it as a
compact graph for offline routing on the Jetson (route.py). Standard library only.

    python osm_build.py                                   # default area: La Paz (centro, Sopocachi, Obrajes, Calacoto)
    python osm_build.py --bbox -16.56,-68.16,-16.48,-68.06 --out ../maps/lapaz_walk.json
    python osm_build.py --from-file overpass.json         # build from a file downloaded elsewhere

This is the only step that needs internet (setup time, like downloading the models). The
navigation itself never goes online. Map data (c) OpenStreetMap contributors, ODbL.

Graph file (JSON):
  nodes: [[lat, lon], ...]         index = node id in the graph
  edges: [[a, b, length_m, kind, name_index], ...]   undirected
  names: ["", "Avenida Hernando Siles", ...]
  kinds: highway type of each edge (footway, residential, steps, ...)
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_BBOX = (-16.565, -68.160, -16.480, -68.055)   # south, west, north, east
SERVERS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
WALK_TYPES = ("footway|pedestrian|path|steps|living_street|residential|unclassified|service|"
              "tertiary|tertiary_link|secondary|secondary_link|primary|primary_link|trunk|trunk_link|"
              "track|corridor|crossing")
EARTH_R = 6371000.0


def query(bbox):
    s, w, n, e = bbox
    return (f'[out:json][timeout:240];'
            f'(way["highway"~"^({WALK_TYPES})$"]({s},{w},{n},{e}););'
            f'(._;>;);out body;')


def download(bbox):
    data = urllib.parse.urlencode({"data": query(bbox)}).encode()
    last = None
    for url in SERVERS:
        try:
            print(f"Downloading from {url} ...")
            t0 = time.time()
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "assistive-navigation-thesis/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read()
            print(f"  {len(raw) / 1e6:.1f} MB in {time.time() - t0:.0f} s")
            return json.loads(raw)
        except Exception as ex:
            print(f"  failed: {ex}")
            last = ex
    sys.exit(f"Could not download the map ({last}). Check the internet connection or try later.")


def walkable(tags):
    if tags.get("highway") not in WALK_TYPES.split("|"):
        return False
    if tags.get("foot") in ("no", "private"):
        return False
    if tags.get("access") in ("private", "no") and tags.get("foot") not in ("yes", "designated", "permissive"):
        return False
    if tags.get("area") == "yes" and tags.get("highway") not in ("pedestrian", "footway"):
        return False
    return True


def dist_m(a, b):
    k = math.pi / 180.0 * EARTH_R
    return math.hypot((b[1] - a[1]) * k * math.cos(math.radians((a[0] + b[0]) / 2)), (b[0] - a[0]) * k)


def build(osm):
    coords, ways = {}, []
    for el in osm.get("elements", []):
        if el["type"] == "node":
            coords[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way":
            ways.append(el)
    use = {}                       # osm node id -> number of walkable ways using it (to find junctions)
    kept = []
    for w in ways:
        tags = w.get("tags", {})
        if not walkable(tags):
            continue
        nds = [n for n in w["nodes"] if n in coords]
        if len(nds) < 2:
            continue
        kept.append((nds, tags))
        for i, n in enumerate(nds):
            use[n] = use.get(n, 0) + (2 if i in (0, len(nds) - 1) else 1)
    # graph nodes: junctions and way ends; intermediate shape points are folded into the edge length
    # but kept as geometry (every ~25 m) so the route can be drawn and turns detected.
    idx, nodes, names, name_idx, edges = {}, [], [""], {"": 0}, []

    def node_id(osm_id):
        if osm_id not in idx:
            idx[osm_id] = len(nodes)
            nodes.append([round(coords[osm_id][0], 7), round(coords[osm_id][1], 7)])
        return idx[osm_id]

    for nds, tags in kept:
        name = tags.get("name", "")
        if name not in name_idx:
            name_idx[name] = len(names)
            names.append(name)
        kind = tags["highway"]
        start, acc, since = nds[0], 0.0, 0.0
        for a, b in zip(nds, nds[1:]):
            d = dist_m(coords[a], coords[b])
            acc += d
            since += d
            is_end = b == nds[-1]
            if use.get(b, 0) > 1 or is_end or since >= 25.0:
                edges.append([node_id(start), node_id(b), round(acc, 1), kind, name_idx[name]])
                start, acc, since = b, 0.0, 0.0
    return {"nodes": nodes, "edges": edges, "names": names,
            "source": "OpenStreetMap contributors (ODbL)", "built": time.strftime("%Y-%m-%d %H:%M")}


def main():
    ap = argparse.ArgumentParser(description="Build the offline walking graph from OpenStreetMap")
    ap.add_argument("--bbox", help="south,west,north,east (default: La Paz urban area)")
    ap.add_argument("--from-file", help="use an Overpass JSON file instead of downloading")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps",
                                                   "lapaz_walk.json"))
    a = ap.parse_args()
    bbox = tuple(float(x) for x in a.bbox.split(",")) if a.bbox else DEFAULT_BBOX
    if a.from_file:
        osm = json.load(open(a.from_file))
    else:
        osm = download(bbox)
        raw_path = os.path.splitext(a.out)[0] + "_overpass.json"
        os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump(osm, f)
        print(f"Raw data kept in {raw_path}")
    g = build(osm)
    g["bbox"] = bbox
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(g, f, separators=(",", ":"), ensure_ascii=False)
    total = sum(e[2] for e in g["edges"]) / 1000
    kinds = {}
    for e in g["edges"]:
        kinds[e[3]] = kinds.get(e[3], 0) + e[2]
    print(f"Saved {a.out}: {len(g['nodes'])} nodes, {len(g['edges'])} edges, {total:.0f} km of walkable ways")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:15s} {v / 1000:7.1f} km")


if __name__ == "__main__":
    main()
