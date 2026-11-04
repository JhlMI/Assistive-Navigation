"""Standalone IPM calibration (when main.py is NOT running).

The same calibration is also available inside the main web view (button "IPM calibration").
Use this script to calibrate from a saved photo (e.g. a snapshot taken on the street).

Usage:
    python calibrate_bev.py                       # live OAK-D Lite camera
    python calibrate_bev.py --image snap.jpg      # a saved 1920x1080 photo
    python calibrate_bev.py --image a.jpg b.jpg c.jpg   # several photos: "Next photo" switches
Each calibration is also appended to bev_calibration_runs.jsonl; compare them with
    python compare_calib.py
Then open http://<jetson-ip>:8081
"""
import argparse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

import config as C
from calib_web import CalibSession, page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", help="calibrate from saved photos instead of the camera")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    camera = None
    grab_label = "Take photo"
    if args.image:
        stills = []
        for path in args.image:
            img = cv2.imread(path)
            if img is None:
                raise SystemExit(f"Could not read {path}")
            stills.append((Path(path).name, cv2.resize(img, (1920, 1080))))
        idx = {"i": -1}

        def provider():                           # each call moves to the next photo
            idx["i"] = (idx["i"] + 1) % len(stills)
            return stills[idx["i"]][1]

        def name():
            return f"{stills[idx['i']][0]} ({idx['i'] + 1}/{len(stills)})"

        session = CalibSession(provider, save_photos=False, name_provider=name)
        grab_label = "Next photo" if len(stills) > 1 else "Reload photo"
        print(f"{len(stills)} photo(s) loaded")
    else:
        from camera import OakCamera
        camera = OakCamera(C.CAMERA_FPS, C.CAMERA_FOCUS, C.FOCUS_RANGE)
        last = {"id": 0, "frame": None}

        def provider():
            for _ in range(10):                   # take a fresh frame
                frame, last["id"] = camera.next(last["id"], timeout=2.0)
                if frame is not None:
                    last["frame"] = frame
            return last["frame"]

        session = CalibSession(provider, save_photos=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, out):
            if out is None:
                self.send_error(404)
                return
            body, ctype = out
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send((page("", grab_label).encode(), "text/html; charset=utf-8"))
            else:
                self._send(session.handle_get(url.path, parse_qs(url.query)))

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            self._send(session.handle_post(urlparse(self.path).path, body))

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Open http://<jetson-ip>:{args.port}  (Ctrl+C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if camera:
            camera.stop()


if __name__ == "__main__":
    main()
