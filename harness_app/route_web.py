"""Web page /route for the macro-navigation (offline map, plan, verify, save, follow).

Served by web.py: handle_get / handle_post return (body, content_type) or None.
The map is drawn from the OSM graph on the Jetson: no internet, no tiles.
"""
import json
from urllib.parse import parse_qs

DEFAULT_CENTER = (-16.527894, -68.106453)      # Plaza Italia (Obrajes)


def _json(obj):
    return json.dumps(obj, ensure_ascii=False).encode(), "application/json"


def _f(q, k, default=None):
    try:
        return float(q[k][0])
    except (KeyError, ValueError, IndexError):
        return default


def handle_get(rm, path, query):
    q = parse_qs(query)
    if path == "/route":
        return PAGE.encode(), "text/html; charset=utf-8"
    if rm is None:
        return _json({"error": "navegación por rutas no disponible"})
    if path == "/route/state":
        return _json(rm.state())
    if path == "/route/map":
        lat, lon = _f(q, "lat", DEFAULT_CENTER[0]), _f(q, "lon", DEFAULT_CENTER[1])
        return _json(rm.map_segments(lat, lon, _f(q, "r", 450.0)))
    if path == "/route/list":
        return _json(rm.list_routes())
    if path == "/route/get":
        return _json(rm.load_route(q.get("file", [""])[0]) or {"error": "no existe"})
    return None


def handle_post(rm, path, query):
    if rm is None:
        return _json({"error": "navegación por rutas no disponible"})
    q = parse_qs(query)
    if path == "/route/plan":
        dest = (_f(q, "lat"), _f(q, "lon"))
        if None in dest:
            return _json({"error": "falta el destino"})
        start = (_f(q, "slat"), _f(q, "slon"))
        return _json(rm.plan_route(dest, None if None in start else start))
    if path == "/route/verify/start":
        return _json(rm.start_verify())
    if path == "/route/verify/corner":
        return _json(rm.mark_corner())
    if path == "/route/verify/hazard":
        return _json(rm.mark_hazard(q.get("kind", ["other"])[0]))
    if path == "/route/verify/finish":
        return _json(rm.finish_verify(q.get("name", [""])[0] or "ruta", q.get("arrived", ["1"])[0] == "1"))
    if path == "/route/verify/cancel":
        return _json(rm.cancel_verify())
    if path == "/route/follow":
        return _json(rm.start_follow(q.get("file", [""])[0]))
    if path == "/route/stop":
        return _json(rm.stop_follow())
    if path == "/route/delete":
        return _json(rm.delete_route(q.get("file", [""])[0]))
    return None


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rutas</title>
<style>
:root{--bg:#111;--card:#1d1d1d;--mut:#999;--ok:#3c8;--warn:#fb3;--bad:#f66;--acc:#f90}
body{margin:0;background:var(--bg);color:#eee;font:15px/1.35 system-ui,sans-serif}
.w{max-width:760px;margin:0 auto;padding:8px 10px 40px}
h2{font-size:16px;margin:14px 0 6px}
.card{background:var(--card);border-radius:8px;padding:10px;margin:8px 0}
button{font:15px system-ui;padding:11px 13px;margin:3px 4px 3px 0;border:0;border-radius:6px;color:#fff;background:#555}
.go{background:#2a7}.blue{background:#357}.red{background:#a33}.org{background:#b60}
input{font:15px system-ui;padding:9px;border-radius:6px;border:1px solid #444;background:#000;color:#eee;width:100%;box-sizing:border-box}
canvas{width:100%;background:#0a0a0a;border-radius:8px;touch-action:none;display:block}
.mut{color:var(--mut);font-size:13px}
#cue{font-size:24px;font-weight:700;text-align:center;padding:12px;border-radius:8px;background:#000;min-height:30px}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:14px}td{padding:5px 3px;border-bottom:1px solid #2a2a2a}
</style></head><body><div class="w">
<div class="row" style="justify-content:space-between"><b>Rutas (sin internet)</b><span id="gps" class="mut"></span></div>
<div id="cue">—</div>
<div id="fol" class="mut"></div>
<canvas id="c" width="760" height="620"></canvas>
<div class="row mut" style="margin-top:4px">
 <button onclick="zoom(1.5)">+</button><button onclick="zoom(1/1.5)">−</button>
 <button onclick="centerMe()">Centrar en mí</button>
 <label><input type="radio" name="tap" value="dest" checked style="width:auto"> tocar = destino</label>
 <label><input type="radio" name="tap" value="start" style="width:auto"> tocar = origen</label>
</div>
<div class="card">
 <div class="mut" id="sel">Toca el mapa para elegir el destino (o pega «lat, lon»).</div>
 <div class="row"><input id="ll" placeholder="-16.52345, -68.11234" style="flex:1"><button class="blue" onclick="useLL()">Usar</button></div>
 <button class="go" onclick="plan()">Calcular ruta</button>
 <div id="pinfo" class="mut"></div>
</div>
<div class="card">
 <h2 style="margin-top:0">Verificar caminando (con acompañante)</h2>
 <div class="mut">Recorre la ruta calculada. En cada giro presiona «Esquina»; marca lo que el mapa no tiene.</div>
 <button class="go" onclick="post('/route/verify/start')">Iniciar verificación</button>
 <button class="org" id="corner" onclick="post('/route/verify/corner')">Esquina</button><br>
 <button onclick="haz('stairs')">Escalera</button><button onclick="haz('step')">Grada</button>
 <button onclick="haz('no_sidewalk')">Sin acera</button><button onclick="haz('crossing')">Cruce</button>
 <button onclick="haz('obstacle')">Obstáculo fijo</button>
 <div class="row" style="margin-top:6px"><input id="rname" placeholder="nombre de la ruta" style="flex:1">
 <button class="go" onclick="finish()">Llegué: guardar</button><button class="red" onclick="post('/route/verify/cancel')">Cancelar</button></div>
 <div id="vinfo" class="mut"></div>
</div>
<div class="card">
 <h2 style="margin-top:0">Rutas verificadas</h2>
 <table id="list"></table>
 <button class="red" onclick="post('/route/stop')">Detener guía de ruta</button>
</div>
<div id="msg" class="mut"></div>
</div><script>
let S=null,seg=[],segC=null,center=null,scale=2.2,dest=null,start=null,fetching=false;
const c=document.getElementById('c'),g=c.getContext('2d');
const K=Math.PI/180*6371000;
const xy=(lat,lon)=>[(lon-center[1])*K*Math.cos(center[0]*Math.PI/180),(lat-center[0])*K];
const P=(lat,lon)=>{const [e,n]=xy(lat,lon);return [c.width/2+e*scale,c.height/2-n*scale];};
const inv=(px,py)=>{const e=(px-c.width/2)/scale,n=(c.height/2-py)/scale;return [center[0]+n/K,center[1]+e/(K*Math.cos(center[0]*Math.PI/180))];};
const KC={footway:'#3a6',pedestrian:'#3a6',path:'#3a6',steps:'#e33',primary:'#a63',trunk:'#a63',secondary:'#986'};
function draw(){
 g.fillStyle='#0a0a0a';g.fillRect(0,0,c.width,c.height);if(!center)return;
 for(const s of seg){const a=P(s[0],s[1]),b=P(s[2],s[3]);g.strokeStyle=KC[s[4]]||'#555';g.lineWidth=s[4]=='steps'?3:1.5;
  g.beginPath();g.moveTo(...a);g.lineTo(...b);g.stroke();}
 const line=(pts,col,w)=>{if(!pts||pts.length<2)return;g.strokeStyle=col;g.lineWidth=w;g.beginPath();pts.forEach((p,i)=>{const q=P(p[0],p[1]);i?g.lineTo(...q):g.moveTo(...q);});g.stroke();};
 const F=S&&S.follow;line(F?F.polyline:(S&&S.plan&&S.plan.polyline),'#f90',4);
 if(S&&S.verify)line(S.verify.track,'#3cf',3);
 const wps=F?F.waypoints:(S&&S.plan?S.plan.waypoints:[]);
 g.font='bold 13px system-ui';
 (wps||[]).forEach((w,i)=>{const q=P(w.lat,w.lon);const col={L:'#fb3',R:'#fb3',STEPS:'#f44',HAZARD:'#f44',END:'#3c8'}[w.turn]||'#fff';
  g.fillStyle=col;g.beginPath();g.arc(...q,F&&i==F.i?9:6,0,7);g.fill();g.fillStyle='#fff';
  g.fillText((i+1)+' '+({L:'izq',R:'der',STEPS:'esc',HAZARD:'!',END:'fin'}[w.turn]||w.turn),q[0]+9,q[1]-8);});
 if(dest){const q=P(...dest);g.strokeStyle='#3c8';g.lineWidth=3;g.beginPath();g.arc(...q,10,0,7);g.stroke();}
 if(start){const q=P(...start);g.strokeStyle='#3cf';g.lineWidth=3;g.strokeRect(q[0]-8,q[1]-8,16,16);}
 if(S&&S.pos){const q=P(S.pos[0],S.pos[1]);g.fillStyle='rgba(255,255,0,.15)';g.beginPath();g.arc(...q,Math.max(6,S.pos[2]*scale),0,7);g.fill();
  g.fillStyle='#ff0';g.beginPath();g.arc(...q,6,0,7);g.fill();
  if(S.heading!=null){const a=S.heading*Math.PI/180;g.strokeStyle='#ff0';g.lineWidth=3;g.beginPath();g.moveTo(...q);g.lineTo(q[0]+28*Math.sin(a),q[1]-28*Math.cos(a));g.stroke();}}
 g.fillStyle='#888';g.font='12px system-ui';g.fillText('N ↑   '+Math.round(c.width/scale)+' m de ancho   © OpenStreetMap',8,c.height-8);
}
async function loadMap(force){
 if(!center||fetching)return;if(!force&&segC&&Math.hypot(...xy(...segC))<180)return;
 fetching=true;try{const r=await (await fetch(`/route/map?lat=${center[0]}&lon=${center[1]}&r=450`)).json();
 if(r.error)msg(r.error);else{seg=r.segments;segC=center.slice();}}catch(e){}fetching=false;draw();}
function zoom(k){scale=Math.min(12,Math.max(0.4,scale*k));draw();}
function centerMe(){if(S&&S.pos){center=[S.pos[0],S.pos[1]];loadMap();draw();}else msg('sin posición GPS');}
let drag=null;
c.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,c:center.slice(),moved:false};});
c.addEventListener('pointermove',e=>{if(!drag)return;const k=c.width/c.getBoundingClientRect().width;const dx=(e.clientX-drag.x)*k,dy=(e.clientY-drag.y)*k;
 if(Math.hypot(dx,dy)>8)drag.moved=true;if(drag.moved){center=[drag.c[0]+dy/scale/K,drag.c[1]-dx/scale/(K*Math.cos(drag.c[0]*Math.PI/180))];draw();}});
c.addEventListener('pointerup',e=>{if(drag&&!drag.moved){const r=c.getBoundingClientRect(),k=c.width/r.width;const ll=inv((e.clientX-r.left)*k,(e.clientY-r.top)*k);
 if(document.querySelector('input[name=tap]:checked').value=='dest')dest=ll;else start=ll;sel();draw();}else if(drag)loadMap();drag=null;});
function sel(){document.getElementById('sel').textContent=(dest?'Destino: '+dest[0].toFixed(6)+', '+dest[1].toFixed(6):'Sin destino')+(start?' | Origen manual: '+start[0].toFixed(6)+', '+start[1].toFixed(6):' | Origen: mi posición');}
function useLL(){const m=document.getElementById('ll').value.match(/-?\\d+\\.\\d+/g);if(!m||m.length<2)return msg('formato: -16.52345, -68.11234');
 dest=[+m[0],+m[1]];center=dest.slice();loadMap(true);sel();draw();}
function msg(t){document.getElementById('msg').textContent=t;}
async function post(u){try{const r=await (await fetch(u,{method:'POST'})).json();msg(r.error?('⚠ '+r.error):JSON.stringify(r));poll();list();return r;}catch(e){msg('sin conexión');}}
async function plan(){if(!dest)return msg('elige el destino');let u=`/route/plan?lat=${dest[0]}&lon=${dest[1]}`;if(start)u+=`&slat=${start[0]}&slon=${start[1]}`;
 const r=await post(u);if(r&&!r.error){const st=r.waypoints.filter(w=>w.turn=='STEPS').length,tu=r.waypoints.filter(w=>w.turn=='L'||w.turn=='R').length;
 document.getElementById('pinfo').innerHTML=`<b>${Math.round(r.length_m)} m</b>, ${tu} giros, ${st} escaleras (${Math.round(r.steps_m)} m)<br>${(r.streets||[]).join(' → ')}`;}}
function haz(k){post('/route/verify/hazard?kind='+k);}
function finish(){const n=document.getElementById('rname').value.trim();if(!n)return msg('ponle un nombre a la ruta');post('/route/verify/finish?name='+encodeURIComponent(n));}
async function list(){try{const L=await (await fetch('/route/list')).json();
 document.getElementById('list').innerHTML=L.length?L.map(r=>`<tr><td><b>${r.name}</b><br><span class="mut">${Math.round(r.length_m||0)} m · ${r.waypoints} puntos · ${r.created||''}</span></td>
 <td style="text-align:right"><button class="go" onclick="post('/route/follow?file=${encodeURIComponent(r.file)}')">Guiar</button>
 <button class="red" onclick="if(confirmDel('${r.name}'))post('/route/delete?file=${encodeURIComponent(r.file)}')">×</button></td></tr>`).join(''):'<tr><td class="mut">Todavía no hay rutas verificadas.</td></tr>';}catch(e){}}
let delArm=null;function confirmDel(n){if(delArm==n){delArm=null;return true;}delArm=n;msg('toca × otra vez para borrar «'+n+'»');return false;}
const CUE={ARRIVED:['#3c8','LLEGASTE'],LOW_GPS:['#777','GPS insuficiente']};
async function poll(){try{S=await (await fetch('/route/state')).json();}catch(e){return;}
 if(!center)center=S.pos?[S.pos[0],S.pos[1]]:(S.plan?S.plan.start:[""" + str(DEFAULT_CENTER[0]) + """,""" + str(DEFAULT_CENTER[1]) + """]);
 loadMap();
 document.getElementById('gps').innerHTML=S.pos?`GPS ${S.gps_good?'<b style="color:#3c8">bueno</b>':'<b style="color:#fb3">débil</b>'} · HDOP ${S.hdop??'-'} · ${S.cn0??'-'} dB-Hz · ±${S.pos[2]} m · rumbo ${S.heading==null?'-':Math.round(S.heading)+'°'}`:'<b style="color:#f66">sin GPS</b>';
 const V=S.verify;document.getElementById('vinfo').textContent=V?`Verificando: esquina ${V.next_corner}/${V.turns} · peligros marcados ${V.hazards.length} · ${V.track.length} puntos grabados`:'';
 document.getElementById('corner').textContent=V?`Esquina (${Math.min(V.next_corner+1,V.turns)}/${V.turns})`:'Esquina';
 const F=S.follow,cue=document.getElementById('cue');
 if(F){const st=await (await fetch('/status')).json().catch(()=>({}));const t=st.cue_text||(F.arrived?'llegaste al destino':'sigue la ruta');
  cue.textContent=t;cue.style.background=st.cue?(st.cue.startsWith('TURN')?'#7a5':st.cue.startsWith('OFF')?'#a52':st.cue=='HAZARD'?'#a22':st.cue=='ARRIVED'?'#284':'#333'):'#000';
  document.getElementById('fol').textContent=`${F.name} · punto ${Math.min(F.i+1,F.waypoints.length)}/${F.waypoints.length}`+(st.wp_dist_m!=null?` · a ${st.wp_dist_m} m · ${st.wp_rel_deg>0?'+':''}${st.wp_rel_deg??'-'}°`:'');}
 else{cue.textContent=V?'verificación en curso':'—';cue.style.background='#000';document.getElementById('fol').textContent='';}
 draw();}
setInterval(poll,1000);poll();list();sel();
</script></body></html>"""
