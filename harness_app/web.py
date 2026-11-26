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
import route_web

CMD_COLOR = {"F": (80, 200, 80), "L": (255, 170, 0), "R": (255, 170, 0),
             "X": (40, 40, 230), "W": (0, 200, 255), "E": (255, 0, 255), "S": (160, 160, 160)}
def _cv_text(img, text, org, scale, color, thickness):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


put_text = _cv_text   # replaced by evidence.py (--lang es) with a renderer that supports accents

CMD_NAME = {"F": "FORWARD", "L": "LEFT", "R": "RIGHT", "X": "STOP", "W": "CAREFUL (no path)",
            "E": "FAULT", "S": "SILENCE"}


class SharedState:
    """Data shared between the main loop and the web server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.walkable = None
        self.result = None
        self.boxes = []
        self.data = {}
        self.guidance = getattr(C, "GUIDANCE_AT_START", False)   # toggled from the web page; OFF at start
        self.recording = False        # toggled from the web page (main loop opens/closes the file)
        self.rec_mode = "ann" if getattr(C, "REC_ANNOTATED", False) else getattr(C, "REC_MODE", "raw")
        self.rec_info = None          # name of the current recording, set by the main loop
        self.snapshots = 0
        self.web_clients = 0          # browsers currently receiving /video (they cost CPU)
        self.gps = None               # gps.Gps or None
        self.geonav = None            # geonav.GeoNav or None (filtered position + compass heading)
        self.route = None             # navroute.RouteManager or None (/route page)

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

    if bev_mode and getattr(res, "stop_zone_px", None):           # STOP zone projected on the camera image
        zp = np.array([(int(u * sx), int(v * sy)) for u, v in res.stop_zone_px], np.int32)
        cv2.polylines(img, [zp], False, (0, 165, 255), 2, cv2.LINE_AA)
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
        put_text(img, f"{name} {score:.2f}", (int(x1 * sx), max(12, int(y1 * sy) - 4)), 0.45, color, 1)

    if res is not None:
        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)

        text = f"{CMD_NAME.get(res.command, res.command)}  ({res.reason})"
        put_text(img, text, (8, 21), 0.6, CMD_COLOR.get(res.command, (255, 255, 255)), 2)
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
 <button id="m" onclick="mode()">Modo: ...</button>
 <button id="r" onclick="toggle('recording')">Recording: ...</button>
 <button id="p" onclick="photo()" style="background:#357">Take photo</button>
 <a href="/calib" target="_blank"><button style="background:#753">IPM calibration</button></a>
 <a href="/clean" target="_blank"><button style="background:#456">Clean view (demo)</button></a>
 <a href="/gps" target="_blank"><button style="background:#375">GPS test</button></a>
 <a href="/route" target="_blank"><button style="background:#a60">Rutas</button></a>
</div>
<div id="msg"></div>
<div id="s"></div>
<script>
async function f(){
 try{const r=await fetch('/status');const d=await r.json();
 document.getElementById('s').innerHTML=Object.entries(d).map(
  ([k,v])=>`<div class="${k=='warning'&&v?'w':''}">${k}: <b>${v??'-'}</b></div>`).join('');}catch(e){}
}
let st={guidance:true,recording:false,rec_mode:'raw'};
function paint(){
 const g=document.getElementById('g'),r=document.getElementById('r');
 g.textContent='Guidance: '+(st.guidance?'ON':'OFF (camera only)');g.style.background=st.guidance?'#2a7':'#a33';
 r.textContent=st.recording?'\u25CF REC (tap to stop)':'Recording: OFF';r.style.background=st.recording?'#c22':'#555';
 const m=document.getElementById('m');m.textContent=st.rec_mode=='ann'?'Modo: inferencia + IPM':'Modo: solo video raw';
 m.style.background=st.recording?'#333':(st.rec_mode=='ann'?'#a5a':'#558');m.disabled=st.recording;}
async function mode(){if(st.recording)return;const r=await fetch('/toggle?rec_mode='+(st.rec_mode=='ann'?'raw':'ann'),{method:'POST'});st=await r.json();paint();}
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


GPS_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GPS test</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:monospace}
.w{max-width:720px;margin:0 auto;padding:10px 12px}
button{font:17px monospace;padding:14px 16px;margin:4px 6px 4px 0;border:0;border-radius:6px;color:#fff;background:#555}
#st{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:6px;margin:8px 0}
#st div{background:#222;padding:6px 8px;border-radius:4px;overflow-wrap:anywhere}
canvas{width:100%;background:#000;border-radius:6px;touch-action:none}
#msg{color:#9cf;min-height:1.2em}
</style></head><body><div class="w">
<h3 style="margin:6px 0">GPS field test</h3>
<button id="t" onclick="test()">Start test</button>
<button id="m" onclick="mark()" style="background:#357">Mark point</button>
<div id="msg"></div>
<div id="st"></div>
<canvas id="c" width="700" height="700"></canvas>
<h4 style="margin:12px 0 4px">Satellites (signal C/N0, dB-Hz)</h4>
<canvas id="k" width="700" height="260"></canvas>
<p style="color:#888">Protocol: start the test, stand still 30 s at corner A and press <b>Mark point</b>,
walk to corner B, wait 30 s and mark, walk back to A, wait 30 s and mark, stop the test.
<b>spread_m</b> = scatter of the last 5 s (read it while standing still).
<b>cn0_top4</b> = mean signal of the 4 strongest satellites: 40 or more is a clear sky; below 35 the sky is
blocked or something near the antenna adds noise (USB 3 cable, Jetson).
This page has no video: it costs nothing to the navigation loop.</p>
</div><script>
let tr={points:[],marks:[],testing:false};
const c=document.getElementById('c'),g=c.getContext('2d');
function draw(){
 const P=tr.points;g.clearRect(0,0,c.width,c.height);
 if(!P.length){g.fillStyle='#888';g.font='18px monospace';g.fillText(tr.testing?'waiting for fixes...':'no test running',20,40);return;}
 let x0=Math.min(...P.map(p=>p[0])),x1=Math.max(...P.map(p=>p[0])),y0=Math.min(...P.map(p=>p[1])),y1=Math.max(...P.map(p=>p[1]));
 const span=Math.max(x1-x0,y1-y0,10)*1.15,cx=(x0+x1)/2,cy=(y0+y1)/2,k=c.width/span;
 const X=e=>c.width/2+(e-cx)*k,Y=n=>c.height/2-(n-cy)*k;
 let step=span>60?20:span>25?10:5;g.strokeStyle='#222';g.lineWidth=1;
 for(let v=Math.floor((cx-span/2)/step)*step;v<cx+span/2;v+=step){g.beginPath();g.moveTo(X(v),0);g.lineTo(X(v),c.height);g.stroke();}
 for(let v=Math.floor((cy-span/2)/step)*step;v<cy+span/2;v+=step){g.beginPath();g.moveTo(0,Y(v));g.lineTo(c.width,Y(v));g.stroke();}
 g.strokeStyle='#2c8';g.lineWidth=2;g.beginPath();P.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();
 const L=P[P.length-1];g.fillStyle='#ff0';g.beginPath();g.arc(X(L[0]),Y(L[1]),6,0,7);g.fill();
 g.font='bold 18px monospace';tr.marks.forEach(m=>{g.fillStyle='#f84';g.beginPath();g.arc(X(m[1]),Y(m[2]),7,0,7);g.fill();g.fillText(m[0],X(m[1])+9,Y(m[2])-9);});
 g.fillStyle='#aaa';g.font='14px monospace';g.fillText('grid '+step+' m   N up   '+P.length+' fixes',10,c.height-12);
}
async function poll(){
 try{tr=await (await fetch('/gps/track')).json();
  const b=document.getElementById('t');b.textContent=tr.testing?'\u25CF Stop test':'Start test';b.style.background=tr.testing?'#c22':'#2a7';
  const keys=['gps_fix','gps_sats','gps_sats_view','gps_cn0_top4','gps_hdop','gps_speed_kmh','spread_m','file',
   'heading_deg','compass_b_ut','nav_sigma_m','nav_speed_kmh','nav_still','nav_rejected'];
  document.getElementById('st').innerHTML=keys.map(k=>`<div>${k}: <b>${tr[k]??'-'}</b></div>`).join('');
  draw();sky();}catch(e){document.getElementById('msg').textContent='no connection';}
}
const kc=document.getElementById('k'),kg=kc.getContext('2d');
const SC={GPS:'#4c9',GLONASS:'#e94',Galileo:'#59f',BeiDou:'#d6d'};
function sky(){
 const S=(tr.sky||[]).slice().sort((a,b)=>(b.snr||0)-(a.snr||0));kg.clearRect(0,0,kc.width,kc.height);
 if(!S.length){kg.fillStyle='#888';kg.font='16px monospace';kg.fillText('no satellite data yet',16,30);return;}
 const H=kc.height-40,bw=Math.min(40,(kc.width-20)/S.length),y=v=>10+H-H*Math.min(v,55)/55;
 kg.strokeStyle='#555';kg.setLineDash([4,4]);[35,40].forEach(v=>{kg.beginPath();kg.moveTo(0,y(v));kg.lineTo(kc.width,y(v));kg.stroke();
  kg.fillStyle='#777';kg.font='12px monospace';kg.fillText(v,kc.width-22,y(v)-3);});kg.setLineDash([]);
 S.forEach((s,i)=>{const x=10+i*bw;kg.fillStyle=SC[s.sys]||'#aaa';
  if(s.snr){kg.fillRect(x,y(s.snr),bw-4,10+H-y(s.snr));kg.fillStyle='#fff';kg.font='11px monospace';kg.fillText(s.snr,x+2,y(s.snr)-3);}
  kg.fillStyle='#aaa';kg.font='10px monospace';kg.fillText(s.prn,x+1,kc.height-16);});
 let lx=10;Object.entries(SC).forEach(([n,c])=>{kg.fillStyle=c;kg.fillRect(lx,kc.height-10,10,8);kg.fillStyle='#aaa';kg.font='11px monospace';kg.fillText(n,lx+13,kc.height-2);lx+=n.length*7+28;});
}
async function test(){const r=await (await fetch('/gps/test?on='+(tr.testing?0:1),{method:'POST'})).json();
 document.getElementById('msg').textContent=r.file?(tr.testing?'Saved: ':'Recording: ')+r.file:(r.error||'');poll();}
async function mark(){const r=await (await fetch('/gps/mark',{method:'POST'})).json();
 document.getElementById('msg').textContent=r.mark?('Marked '+r.mark):'Start the test first';}
setInterval(poll,1000);poll();
</script></body></html>"""


def start_server(state):
    calib = CalibSession(lambda: state.frame, save_photos=True)

    def toggles():
        return json.dumps({"guidance": state.guidance, "recording": state.recording,
                           "rec_mode": state.rec_mode}).encode()

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
            elif url.path == "/route" or url.path.startswith("/route/"):
                out = route_web.handle_get(state.route, url.path, url.query)
                if out:
                    self._send(*out)
                else:
                    self.send_error(404)
            elif url.path == "/gps":
                self._send(GPS_PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/gps/track":
                body = state.gps.track() if state.gps else {"points": [], "marks": [], "testing": False,
                                                            "gps_fix": "GPS not connected"}
                if state.geonav is not None:
                    try:
                        body.update(state.geonav.status())
                    except Exception:
                        pass
                self._send(json.dumps(body).encode(), "application/json")
            elif self.path == "/clean":
                self._send(CLEAN_PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/video":
                clean = url.query == "clean=1"        # raw camera only, smoother (demo screen)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                period = 1.0 / (C.CLEAN_FPS if clean else C.WEB_FPS)
                last = None
                state.web_clients += 1
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
                finally:
                    state.web_clients -= 1
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
                if "rec_mode" in q and q["rec_mode"][0] in ("raw", "ann") and not state.recording:
                    state.rec_mode = q["rec_mode"][0]      # only while not recording
                if "recording" in q:
                    state.recording = q["recording"][0] == "1"
                self._send(toggles(), "application/json")
            elif url.path.startswith("/route/"):
                out = route_web.handle_post(state.route, url.path, url.query)
                if out:
                    self._send(*out)
                else:
                    self.send_error(404)
            elif url.path in ("/gps/test", "/gps/mark"):
                gps = state.gps
                if gps is None:
                    out = {"error": "GPS not connected"}
                elif url.path == "/gps/mark":
                    out = {"mark": gps.mark()}
                elif parse_qs(url.query).get("on", ["1"])[0] == "1":
                    out = {"file": gps.start_test().name}
                else:
                    path = gps.stop_test()
                    out = {"file": path.name if path else None}
                self._send(json.dumps(out).encode(), "application/json")
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
