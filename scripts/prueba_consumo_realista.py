"""
Prueba de consumo REALISTA del pipeline de percepcion en el Jetson Orin Nano.

A diferencia de la prueba de estres (CPU+GPU al 100 %), aqui se ejecuta el pipeline
elegido para el proyecto tal como correria en el chaleco:

    camara OAK-D Lite 1080p (o imagenes de test en bucle)
      -> YOLO26n-sem 512x896 TensorRT FP16  (segmentacion)
      -> filtro de acera transitable (componente conectada al borde inferior)
      -> YOLOv8n 384x640 TensorRT FP16      (deteccion) + NMS
    limitado a la tasa de la camara (30 FPS por defecto)

Fases (todas registradas con tegrastats cada 500 ms):
    1. reposo   : sistema sin carga (linea base)                  -> --reposo  s
    2. carga    : carga de engines + 5 s de calentamiento (no se resume)
    3. pipeline : pipeline completo a tasa real                    -> --duracion s

NO usa jetson_clocks ni ventilador fijo: el gobernador DVFS actua como en uso real.
NO importa ultralytics (dejaria OpenCV en 1 hilo).

Uso:
    python prueba_consumo_realista.py --fuente camara
    python prueba_consumo_realista.py --fuente imagenes
    python prueba_consumo_realista.py --fuente camara --duracion 600 --fps 15
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BASE = Path.home() / "Assistive-Navigation"
DIR_TEST = BASE / "data" / "test_lapaz" / "test"
ENGINE_SEG = BASE / "modelos" / "segmentacion" / "engines" / "yolo26n_sem_512x896_fp16.engine"
ENGINE_DET = BASE / "modelos" / "deteccion" / "engines" / "yolov8n_384x640_fp16.engine"
DIR_LOGS = BASE / "logs" / "consumo_realista"

ID_ACERA = 4            # [background, building, pole, road, sidewalk, vegetation]
AREA_MIN_ACERA = 120    # px en la mascara a resolucion del modelo (~500 px en 1080p)
CALENTAMIENTO_S = 5

REGEX_TEGRA = {
    "vdd_in_mw": re.compile(r"VDD_IN (\d+)mW"),
    "vdd_cpu_gpu_cv_mw": re.compile(r"VDD_CPU_GPU_CV (\d+)mW"),
    "vdd_soc_mw": re.compile(r"VDD_SOC (\d+)mW"),
    "ram_mb": re.compile(r"RAM (\d+)/\d+MB"),
    "gpu_pct": re.compile(r"GR3D_FREQ (\d+)%"),
    "temp_tj_c": re.compile(r"tj@([\d.]+)C"),
}
REGEX_CPU = re.compile(r"CPU \[([^\]]+)\]")
REGEX_CPU_NUCLEO = re.compile(r"(\d+)%@(\d+)")


# ---------------------------------------------------------------------------
# tegrastats en segundo plano, con etiqueta de fase
# ---------------------------------------------------------------------------
class Tegrastats:
    def __init__(self, ruta_log, intervalo_ms=500):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(intervalo_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self.archivo = open(ruta_log, "w")
        self.fase = "inicio"
        self.muestras = {}
        self.ultima = {}
        self.hilo = threading.Thread(target=self._leer, daemon=True)
        self.hilo.start()

    def _leer(self):
        for linea in self.proc.stdout:
            fase = self.fase
            self.archivo.write(f"[{fase}] {linea}")
            self.archivo.flush()
            muestra = {}
            for clave, regex in REGEX_TEGRA.items():
                m = regex.search(linea)
                if m:
                    muestra[clave] = float(m.group(1))
            m = REGEX_CPU.search(linea)
            if m:
                nucleos = REGEX_CPU_NUCLEO.findall(m.group(1))
                if nucleos:
                    muestra["cpu_pct"] = float(np.mean([float(p) for p, _ in nucleos]))
                    muestra["cpu_mhz"] = float(np.mean([float(f) for _, f in nucleos]))
            if muestra:
                self.muestras.setdefault(fase, []).append(muestra)
                self.ultima = muestra

    def cambiar_fase(self, fase):
        self.fase = fase

    def cerrar(self):
        self.proc.terminate()
        time.sleep(0.6)
        self.archivo.close()


def resumir(muestras):
    resumen = {"n_muestras": len(muestras)}
    claves = list(REGEX_TEGRA) + ["cpu_pct", "cpu_mhz"]
    for clave in claves:
        valores = np.array([m[clave] for m in muestras if clave in m])
        if len(valores):
            resumen[clave] = {
                "media": float(valores.mean()),
                "p95": float(np.percentile(valores, 95)),
                "max": float(valores.max()),
                "min": float(valores.min()),
            }
    return resumen


# ---------------------------------------------------------------------------
# Motor TensorRT (igual que en los scripts de evaluacion)
# ---------------------------------------------------------------------------
class MotorTRT:
    def __init__(self, ruta_engine):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.ERROR)
        with open(ruta_engine, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.contexto = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            nombre = self.engine.get_tensor_name(i)
            forma = tuple(self.engine.get_tensor_shape(nombre))
            np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(nombre)))
            torch_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
            buffer = torch.empty(forma, dtype=torch_dtype, device="cuda")
            self.buffers[nombre] = buffer
            self.contexto.set_tensor_address(nombre, buffer.data_ptr())
            if self.engine.get_tensor_mode(nombre) == trt.TensorIOMode.INPUT:
                self.entrada = nombre
            else:
                self.salida = nombre

    def __call__(self, x):
        with self.torch.cuda.stream(self.stream):
            self.buffers[self.entrada].copy_(self.torch.from_numpy(x))
            self.contexto.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buffers[self.salida].cpu().numpy()


# ---------------------------------------------------------------------------
# Pre y postprocesamiento
# ---------------------------------------------------------------------------
def letterbox(img_bgr, h, w):
    h0, w0 = img_bgr.shape[:2]
    r = min(h / h0, w / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    dw, dh = (w - nw) / 2, (h - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    x = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, left, top, nh, nw


def post_segmentacion(mascara, h, w, left, top, nh, nw):
    """Recorta el padding y deja la mascara a resolucion del modelo (504x896 para 1080p).
    No se reescala a 1080p: el filtro de acera y el IPM no necesitan esa resolucion."""
    return np.asarray(mascara).reshape(h, w).astype(np.uint8)[top:top + nh, left:left + nw]


ETAPAS = ["pre_seg", "inf_seg", "post_seg_filtro", "pre_det", "inf_det", "post_det_nms"]
KERNEL_APERTURA = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def filtrar_acera(mascara):
    """Acera transitable = componentes de 'sidewalk' que tocan el borde inferior."""
    acera = (mascara == ID_ACERA).astype(np.uint8)
    acera = cv2.morphologyEx(acera, cv2.MORPH_OPEN, KERNEL_APERTURA)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(acera, connectivity=8)
    if n <= 1:
        return np.zeros_like(acera)
    ultima_fila = etiquetas[-1]
    validas = [i for i in np.unique(ultima_fila) if i != 0 and stats[i, cv2.CC_STAT_AREA] >= AREA_MIN_ACERA]
    if not validas:
        return np.zeros_like(acera)
    return np.isin(etiquetas, validas).astype(np.uint8)


def post_deteccion(salida, r, pad_x, pad_y, forma_original, conf_min=0.25, iou_nms=0.45):
    pred = salida[0].T
    scores_clase = pred[:, 4:]
    clases = scores_clase.argmax(axis=1)
    scores = scores_clase[np.arange(len(pred)), clases]
    keep = scores >= conf_min
    if not keep.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    pred, scores, clases = pred[keep], scores[keep], clases[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    offset = clases * 4096.0
    cajas_nms = np.stack([cx - bw / 2 + offset, cy - bh / 2 + offset, bw, bh], axis=1)
    indices = cv2.dnn.NMSBoxes(cajas_nms.tolist(), scores.tolist(), conf_min, iou_nms)
    indices = np.array(indices).flatten() if len(indices) else np.array([], dtype=int)
    cajas = np.stack([(cx - bw / 2 - pad_x) / r, (cy - bh / 2 - pad_y) / r,
                      (cx + bw / 2 - pad_x) / r, (cy + bh / 2 - pad_y) / r], axis=1)
    alto, ancho = forma_original
    cajas[:, [0, 2]] = cajas[:, [0, 2]].clip(0, ancho)
    cajas[:, [1, 3]] = cajas[:, [1, 3]].clip(0, alto)
    return cajas[indices], scores[indices], clases[indices]


def construir_procesador():
    seg = MotorTRT(ENGINE_SEG)
    det = MotorTRT(ENGINE_DET)

    def procesar(img):
        reloj = time.perf_counter
        t = [reloj()]
        x, _, l, tp, nh, nw = letterbox(img, 512, 896);                 t.append(reloj())
        salida_seg = seg(x);                                            t.append(reloj())
        mascara = post_segmentacion(salida_seg, 512, 896, l, tp, nh, nw)
        transitable = filtrar_acera(mascara);                           t.append(reloj())
        xd, rd, ld, td, _, _ = letterbox(img, 384, 640);                t.append(reloj())
        salida_det = det(xd);                                           t.append(reloj())
        cajas, _, _ = post_deteccion(salida_det, rd, ld, td, img.shape[:2]); t.append(reloj())
        tiempos = {etapa: (t[k + 1] - t[k]) * 1000 for k, etapa in enumerate(ETAPAS)}
        return transitable, cajas, tiempos

    return procesar


# ---------------------------------------------------------------------------
# Fuentes de imagen
# ---------------------------------------------------------------------------
def abrir_camara(fps):
    import depthai as dai

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam.setIspScale(1, 2)
    cam.setVideoSize(1920, 1080)
    cam.setFps(fps)
    cam.initialControl.setSharpness(0)
    cam.initialControl.setLumaDenoise(0)
    cam.initialControl.setChromaDenoise(4)
    cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    cam.initialControl.setAutoFocusLensRange(50, 70)
    cola = cam.video.createOutputQueue(maxSize=2, blocking=False)
    pipeline.start()
    return pipeline, cola


def modo_energia():
    try:
        salida = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
        return " ".join(salida.split())
    except Exception:
        return "desconocido"


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuente", choices=["camara", "imagenes"], required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duracion", type=int, default=300, help="s de pipeline medido")
    parser.add_argument("--reposo", type=int, default=30, help="s de linea base")
    args = parser.parse_args()

    if "ultralytics" in sys.modules:
        sys.exit("ultralytics esta importado: abortar (deja OpenCV en 1 hilo)")

    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_raw = DIR_LOGS / f"tegrastats_{args.fuente}_{ts}.log"
    ruta_json = DIR_LOGS / f"resumen_{args.fuente}_{ts}.json"

    print(f"Modo de energia : {modo_energia()}")
    print(f"Hilos OpenCV    : {cv2.getNumThreads()}")
    print(f"Fuente          : {args.fuente} | FPS objetivo {args.fps} | {args.duracion} s")
    print(f"Log crudo       : {ruta_raw}\n")

    tegra = Tegrastats(ruta_raw)

    # 1. Reposo ---------------------------------------------------------------
    print(f"[1/3] Reposo {args.reposo} s (no toques nada)...")
    time.sleep(2)
    tegra.cambiar_fase("reposo")
    time.sleep(args.reposo)

    # 2. Carga ----------------------------------------------------------------
    tegra.cambiar_fase("carga")
    print("[2/3] Cargando engines y fuente...")
    procesar = construir_procesador()
    pipeline_cam, cola = None, None
    imagenes = []
    if args.fuente == "camara":
        pipeline_cam, cola = abrir_camara(args.fps)
    else:
        imagenes = [cv2.imread(str(p)) for p in sorted(DIR_TEST.glob("*.jpg"))]
        if not imagenes:
            sys.exit(f"No hay imagenes en {DIR_TEST}")

    periodo = 1.0 / args.fps

    def obtener_frame(i):
        if pipeline_cam is not None:
            return cola.get().getCvFrame()   # la camara marca el ritmo
        return imagenes[i % len(imagenes)]

    fin = time.time() + CALENTAMIENTO_S
    i = 0
    while time.time() < fin:
        procesar(obtener_frame(i))  # descarta tiempos del calentamiento
        i += 1

    # 3. Pipeline -------------------------------------------------------------
    print(f"[3/3] Pipeline realista {args.duracion} s (Ctrl+C para cortar antes)\n")
    tegra.cambiar_fase("pipeline")
    latencias, n_frames, n_cajas, frac_acera = [], 0, [], []
    t_etapas = {e: [] for e in ETAPAS + ["espera_frame"]}
    t0 = time.perf_counter()
    proximo = t0
    ultimo_print = t0
    try:
        while time.perf_counter() - t0 < args.duracion:
            ts_frame = time.perf_counter()
            frame = obtener_frame(n_frames)
            ts_ini = time.perf_counter()
            transitable, cajas, tiempos = procesar(frame)
            latencias.append((time.perf_counter() - ts_ini) * 1000)
            t_etapas["espera_frame"].append((ts_ini - ts_frame) * 1000)
            for e, v in tiempos.items():
                t_etapas[e].append(v)
            n_cajas.append(len(cajas))
            frac_acera.append(float(transitable.mean()))
            n_frames += 1

            if pipeline_cam is None:             # imagenes: limitar a la tasa de la camara
                proximo += periodo
                espera = proximo - time.perf_counter()
                if espera > 0:
                    time.sleep(espera)
                else:
                    proximo = time.perf_counter()

            ahora = time.perf_counter()
            if ahora - ultimo_print >= 10:
                fps = n_frames / (ahora - t0)
                u = tegra.ultima
                print(f"  t={ahora - t0:5.0f}s | FPS {fps:5.1f} | lat med {np.median(latencias[-300:]):5.1f} ms | "
                      f"VDD_IN {u.get('vdd_in_mw', 0) / 1000:5.2f} W | CPU {u.get('cpu_pct', 0):4.0f}% | "
                      f"GPU {u.get('gpu_pct', 0):3.0f}% | Tj {u.get('temp_tj_c', 0):4.1f} C")
                ultimo_print = ahora
    except KeyboardInterrupt:
        print("\nCortado por el usuario.")
    transcurrido = time.perf_counter() - t0

    tegra.cambiar_fase("fin")
    if pipeline_cam is not None:
        pipeline_cam.stop()
    tegra.cerrar()

    # Resumen -----------------------------------------------------------------
    rep = resumir(tegra.muestras.get("reposo", []))
    pip = resumir(tegra.muestras.get("pipeline", []))
    lat = np.array(latencias)
    fps_real = n_frames / transcurrido

    resultado = {
        "fecha": ts, "fuente": args.fuente, "fps_objetivo": args.fps,
        "modo_energia": modo_energia(), "hilos_opencv": cv2.getNumThreads(),
        "duracion_s": transcurrido, "frames": n_frames, "fps": fps_real,
        "latencia_mediana_ms": float(np.median(lat)),
        "latencia_p95_ms": float(np.percentile(lat, 95)),
        "latencia_p99_ms": float(np.percentile(lat, 99)),
        "latencia_max_ms": float(lat.max()),
        "cajas_media": float(np.mean(n_cajas)),
        "fraccion_acera_media": float(np.mean(frac_acera)),
        "etapas_mediana_ms": {e: float(np.median(v)) for e, v in t_etapas.items() if v},
        "etapas_p99_ms": {e: float(np.percentile(v, 99)) for e, v in t_etapas.items() if v},
        "reposo": rep, "pipeline": pip,
    }
    if "vdd_in_mw" in rep and "vdd_in_mw" in pip:
        p_rep, p_pip = rep["vdd_in_mw"]["media"], pip["vdd_in_mw"]["media"]
        resultado["potencia_incremental_mw"] = p_pip - p_rep
        resultado["energia_total_mj_por_frame"] = p_pip / fps_real
        resultado["energia_incremental_mj_por_frame"] = (p_pip - p_rep) / fps_real
        resultado["energia_wh_por_hora"] = p_pip / 1000

    with open(ruta_json, "w") as f:
        json.dump(resultado, f, indent=2)

    def fila(nombre, clave, escala=1.0, fmt="{:8.2f}", stat="media"):
        a = rep.get(clave, {}).get(stat)
        b = pip.get(clave, {}).get(stat)
        sa = fmt.format(a * escala) if a is not None else "     n/a"
        sb = fmt.format(b * escala) if b is not None else "     n/a"
        print(f"  {nombre:24s} {sa:>10s} {sb:>10s}")

    print("\n================ RESUMEN ================")
    print(f"  FPS real {fps_real:.1f} | latencia mediana {resultado['latencia_mediana_ms']:.1f} ms | "
          f"p99 {resultado['latencia_p99_ms']:.1f} ms | max {resultado['latencia_max_ms']:.1f} ms")
    print("\n  Tiempo por etapa (mediana / p99, ms):")
    for e in ["espera_frame"] + ETAPAS:
        print(f"    {e:18s} {resultado['etapas_mediana_ms'][e]:6.2f} / {resultado['etapas_p99_ms'][e]:6.2f}")
    print(f"\n  {'':24s} {'reposo':>10s} {'pipeline':>10s}")
    fila("VDD_IN media (W)", "vdd_in_mw", 1 / 1000)
    fila("VDD_IN max (W)", "vdd_in_mw", 1 / 1000, stat="max")
    fila("CPU+GPU+CV media (W)", "vdd_cpu_gpu_cv_mw", 1 / 1000)
    fila("SOC media (W)", "vdd_soc_mw", 1 / 1000)
    fila("CPU media (%)", "cpu_pct", fmt="{:8.0f}")
    fila("CPU frecuencia (MHz)", "cpu_mhz", fmt="{:8.0f}")
    fila("GPU media (%)", "gpu_pct", fmt="{:8.0f}")
    fila("Tj max (C)", "temp_tj_c", fmt="{:8.1f}", stat="max")
    fila("RAM max (MB)", "ram_mb", fmt="{:8.0f}", stat="max")
    if "potencia_incremental_mw" in resultado:
        print(f"\n  Potencia incremental     : {resultado['potencia_incremental_mw'] / 1000:.2f} W")
        print(f"  Energia por frame        : {resultado['energia_total_mj_por_frame']:.1f} mJ total | "
              f"{resultado['energia_incremental_mj_por_frame']:.1f} mJ incremental")
        print(f"  Consumo estimado         : {resultado['energia_wh_por_hora']:.2f} Wh por hora (solo Jetson + camara)")
    print(f"\n  JSON: {ruta_json}")


if __name__ == "__main__":
    main()
