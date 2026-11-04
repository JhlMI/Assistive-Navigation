"""Live web view (standard library only).

http://<jetson-ip>:8080  -> annotated image + status (battery, FPS, command).
The image is annotated and compressed only while someone is watching, at WEB_FPS.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import config as C
from calib_web import CalibSession, page as calib_page, save_snapshot
from planner import render_grid

CMD_COLOR = {"F": (80, 200, 80), "L": (255, 170, 0), "R": (255, 170, 0),
             "X": (40, 40, 230), "S": (160, 160, 160)}
CMD_NAME = {"F": "FORWARD", "L": "LEFT", "R": "RIGHT", "X": "STOP", "S": "SILENCE"}


class SharedState:
    """Data shared between the main loop and the web server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.walkable = None
        self.result = None
        self.boxes = []
        self.data = {}
        self.guidance = True          # toggled from the web page
        self.recording = False        # toggled from the web page (main loop opens/closes the file)
        self.rec_info = None          # name of the current recording, set by the main loop
        self.snapshots = 0

    def update(self, frame, walkable, boxes, result, data):
        with self.lock:
            self.frame, self.walkable, self.boxes = frame, walkable, boxes
            self.result, self.data = result, data

    def snapshot(self):
        with self.lock:
            return self.frame, self.walkable, self.boxes, self.result, dict(self.data)


def annotate(frame, walkable, boxes, res, size=None):
    """Camera image with the perception overlays (+ the bird's-eye panel in BEV mode).
    size: (width, height) of the camera part; default = the live web view size."""
    W, H = size or (C.WEB_WIDTH, C.WEB_HEIGHT)
    img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    sx, sy = W / frame.shape[1], H / frame.shape[0]

    if res is not None and res.reason == "guidance off":      # camera only, no overlays
        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(img, "GUIDANCE OFF (camera only)", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 80, 255), 2, cv2.LINE_AA)
        return img

    bev_mode = res is not None and hasattr(res, "path_px")

    if walkable is not None:
        m = cv2.resize(walkable, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        green = np.zeros_like(img)
        green[:] = (0, 200, 0)
        img[m] = cv2.addWeighted(img, 0.55, green, 0.45, 0)[m]
        if not bev_mode:
            y_roi = int(H * C.ROI_START)
            cv2.line(img, (0, y_roi), (W, y_roi), (255, 255, 255), 1)
            if res is not None and res.center_x is not None:
                cx = int(res.center_x * W / walkable.shape[1])
                cv2.line(img, (cx, y_roi), (cx, H), (0, 255, 255), 2)

    if bev_mode:
        pts = [(int(u * sx), int(v * sy)) for u, v in res.path_px if 0 <= v < frame.shape[0] * 1.2]
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 230, 255), 3)
    else:
        for xc in C.CORRIDOR:
            cv2.line(img, (int(W * xc), 0), (int(W * xc), H), (200, 200, 200), 1)

    obst = {tuple(o[:4]) for o in (res.obstacles if res else [])}
    for (x1, y1, x2, y2, score, cls) in boxes:
        color = (0, 0, 255) if (x1, y1, x2, y2) in obst else (255, 120, 0)
        cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), color, 2)
        name = C.OBSTACLE_CLASSES.get(cls, str(cls))
        cv2.putText(img, f"{name} {score:.2f}", (int(x1 * sx), max(12, int(y1 * sy) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if res is not None:
        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)

        text = f"{CMD_NAME.get(res.command, res.command)}  ({res.reason})"
        cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    CMD_COLOR.get(res.command, (255, 255, 255)), 2, cv2.LINE_AA)
    if bev_mode:
        panel = render_grid(res, H)
        if panel is not None:
            img = np.hstack([img, panel])
    return img


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Assisted navigation</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:monospace}
img{width:100%;max-width:1320px;display:block;margin:0 auto}
#s{max-width:960px;margin:8px auto;padding:0 12px;display:grid;
grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:6px}
#s div{background:#222;padding:6px 8px;border-radius:4px}
.w{color:#f66}
#b,#msg{max-width:960px;margin:8px auto;padding:0 12px}
#b button{background:#555;font:15px monospace;padding:10px 14px;margin:3px 4px 3px 0;border:0;border-radius:4px;color:#fff;cursor:pointer}
#msg{color:#9cf;min-height:1em}
</style></head><body>
<img src="/video">
<div id="b">
 <button id="g" onclick="toggle('guidance')">Guidance: ...</button>
 <button id="r" onclick="toggle('recording')">Recording: ...</button>
 <button id="p" onclick="photo()" style="background:#357">Take photo</button>
 <a href="/calib" target="_blank"><button style="background:#753">IPM calibration</button></a>
 <a href="/clean" target="_blank"><button style="background:#456">Clean view (demo)</button></a>
</div>
<div id="msg"></div>
<div id="s"></div>
<script>
async function f(){
 try{const r=await fetch('/status');const d=await r.json();
 document.getElementById('s').innerHTML=Object.entries(d).map(
  ([k,v])=>`<div class="${k=='warning'&&v?'w':''}">${k}: <b>${v??'-'}</b></div>`).join('');}catch(e){}
}
let st={guidance:true,recording:false};
function paint(){
 const g=document.getElementById('g'),r=document.getElementById('r');
 g.textContent='Guidance: '+(st.guidance?'ON':'OFF (camera only)');g.style.background=st.guidance?'#2a7':'#a33';
 r.textContent=st.recording?'\u25CF REC (tap to stop)':'Recording: OFF';r.style.background=st.recording?'#c22':'#555';}
async function toggle(k){const r=await fetch('/toggle?'+k+'='+(st[k]?0:1),{method:'POST'});st=await r.json();paint();}
async function sync(){try{st=await (await fetch('/toggle')).json();paint();}catch(e){}}
async function photo(){const r=await (await fetch('/photo',{method:'POST'})).json();
 document.getElementById('msg').textContent=r.saved?('Photo saved: '+r.saved):'No frame yet';}
setInterval(f,1000);setInterval(sync,2000);f();sync();
</script></body></html>"""


CLEAN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Demo</title>
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden}
img{width:100vw;height:100vh;object-fit:contain;display:block}</style></head>
<body><img src="/video?clean=1" ondblclick="document.documentElement.requestFullscreen()"
 title="double click: full screen"></body></html>"""


def start_server(state):
    calib = CalibSession(lambda: state.frame, save_photos=True)

    def toggles():
        return json.dumps({"guidance": state.guidance, "recording": state.recording}).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if self.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/toggle":
                self._send(toggles(), "application/json")
            elif url.path == "/calib":
                self._send(calib_page("/calib").encode(), "text/html; charset=utf-8")
            elif url.path.startswith("/calib/"):
                out = calib.handle_get(url.path[len("/calib"):], parse_qs(url.query))
                if out:
                    self._send(*out)
                else:
                    self.send_error(404)
            elif self.path == "/status":
                self._send(json.dumps(state.snapshot()[4]).encode(), "application/json")
            elif self.path == "/clean":
                self._send(CLEAN_PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/video":
                clean = url.query == "clean=1"        # raw camera only, smoother (demo screen)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                period = 1.0 / (C.CLEAN_FPS if clean else C.WEB_FPS)
                last = None
                try:
                    while True:
                        t = time.monotonic()
                        frame, walkable, boxes, res, _ = state.snapshot()
                        if frame is not None and not (clean and frame is last):
                            last = frame
                            if clean:
                                h = int(C.CLEAN_WIDTH * frame.shape[0] / frame.shape[1])
                                img = cv2.resize(frame, (C.CLEAN_WIDTH, h), interpolation=cv2.INTER_AREA)
                            else:
                                img = annotate(frame, walkable, boxes, res)
                            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80 if clean else 70])
                            if ok:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                                 + jpg.tobytes() + b"\r\n")
                        time.sleep(max(0.0, period - (time.monotonic() - t)))
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            url = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            if url.path == "/toggle":
                q = parse_qs(url.query)
                if "guidance" in q:
                    state.guidance = q["guidance"][0] == "1"
                    print(f"Guidance {'ON' if state.guidance else 'OFF'} (from web)")
                if "recording" in q:
                    state.recording = q["recording"][0] == "1"
                self._send(toggles(), "application/json")
            elif url.path == "/photo":
                frame = state.frame
                saved = str(save_snapshot(frame)) if frame is not None else None
                if saved:
                    state.snapshots += 1
                    print(f"Photo saved: {saved}")
                self._send(json.dumps({"saved": saved}).encode(), "application/json")
            elif url.path.startswith("/calib/"):
                out = calib.handle_post(url.path[len("/calib"):], body)
                if out:
                    self._send(*out)
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("0.0.0.0", C.WEB_PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
