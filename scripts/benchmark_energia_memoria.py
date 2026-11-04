"""
Benchmark de energia y memoria del pipeline de percepcion en el Jetson Orin Nano.

Configuraciones (cada una en un proceso separado para que la memoria no se contamine):
    idle -> sistema en reposo (linea base de potencia y RAM)
    seg  -> solo segmentacion YOLO26n-sem 512x896
    A    -> segmentacion 512x896 + detector YOLOv8n 384x640 (preprocesamiento separado)
    B    -> segmentacion + detector YOLOv8n, ambos 512x896 compartiendo el mismo tensor

Mide durante N segundos de inferencia continua (pipeline completo: pre + inferencia + post):
    - FPS y latencia por frame (mediana, p99)
    - Potencia total VDD_IN y VDD_CPU_GPU_CV (tegrastats), energia por frame
    - RAM del sistema (tegrastats) y RSS del proceso
    - Uso de GPU y temperatura

Uso:
    python benchmark_energia_memoria.py --config idle
    python benchmark_energia_memoria.py --config seg
    python benchmark_energia_memoria.py --config A
    python benchmark_energia_memoria.py --config B
    python benchmark_energia_memoria.py --resumen
"""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path.home() / "Assistive-Navigation"
DIR_TEST = BASE / "data" / "test_lapaz" / "test"
ENGINE_SEG = BASE / "modelos" / "segmentacion" / "engines" / "yolo26n_sem_512x896_fp16.engine"
ENGINE_DET = {
    (384, 640): BASE / "modelos" / "deteccion" / "engines" / "yolov8n_384x640_fp16.engine",
    (512, 896): BASE / "modelos" / "deteccion" / "engines" / "yolov8n_512x896_fp16.engine",
}
DIR_LOGS = BASE / "logs" / "benchmark_energia"
CONFIGS = ["idle", "seg", "A", "B"]

REGEX_TEGRA = {
    "vdd_in_mw": re.compile(r"VDD_IN (\d+)mW"),
    "vdd_cpu_gpu_cv_mw": re.compile(r"VDD_CPU_GPU_CV (\d+)mW"),
    "vdd_soc_mw": re.compile(r"VDD_SOC (\d+)mW"),
    "ram_mb": re.compile(r"RAM (\d+)/\d+MB"),
    "gpu_pct": re.compile(r"GR3D_FREQ (\d+)%"),
    "temp_tj_c": re.compile(r"tj@([\d.]+)C"),
}


# ---------------------------------------------------------------------------
# tegrastats en segundo plano
# ---------------------------------------------------------------------------
class Tegrastats:
    def __init__(self, intervalo_ms=200):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(intervalo_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self.muestras = []
        self.midiendo = False
        self.hilo = threading.Thread(target=self._leer, daemon=True)
        self.hilo.start()

    def _leer(self):
        for linea in self.proc.stdout:
            if not self.midiendo:
                continue
            muestra = {}
            for clave, regex in REGEX_TEGRA.items():
                m = regex.search(linea)
                if m:
                    muestra[clave] = float(m.group(1))
            if muestra:
                self.muestras.append(muestra)

    def iniciar(self):
        self.muestras = []
        self.midiendo = True

    def detener(self):
        self.midiendo = False
        return list(self.muestras)

    def cerrar(self):
        self.proc.terminate()


def resumir_tegra(muestras):
    resumen = {"n_muestras": len(muestras)}
    for clave in REGEX_TEGRA:
        valores = np.array([m[clave] for m in muestras if clave in m])
        if len(valores):
            resumen[clave] = {
                "media": float(valores.mean()),
                "p95": float(np.percentile(valores, 95)),
                "max": float(valores.max()),
            }
    return resumen


def leer_rss_mb():
    with open("/proc/self/status") as f:
        for linea in f:
            if linea.startswith("VmRSS:"):
                return int(linea.split()[1]) / 1024
    return None


# ---------------------------------------------------------------------------
# Motor TensorRT
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
# Pre y postprocesamiento (identicos a los scripts de evaluacion)
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


def post_segmentacion(mascara, h, w, left, top, nh, nw, forma_original):
    m = np.asarray(mascara).reshape(h, w).astype(np.uint8)[top:top + nh, left:left + nw]
    alto, ancho = forma_original
    return cv2.resize(m, (ancho, alto), interpolation=cv2.INTER_NEAREST)


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


def construir_pipeline(config):
    seg = MotorTRT(ENGINE_SEG)

    if config == "seg":
        def paso(img):
            x, r, l, t, nh, nw = letterbox(img, 512, 896)
            return post_segmentacion(seg(x), 512, 896, l, t, nh, nw, img.shape[:2])
        return paso

    if config == "A":
        det = MotorTRT(ENGINE_DET[(384, 640)])

        def paso(img):
            x, r, l, t, nh, nw = letterbox(img, 512, 896)
            mascara = post_segmentacion(seg(x), 512, 896, l, t, nh, nw, img.shape[:2])
            xd, rd, ld, td, _, _ = letterbox(img, 384, 640)
            cajas = post_deteccion(det(xd), rd, ld, td, img.shape[:2])
            return mascara, cajas
        return paso

    if config == "B":
        det = MotorTRT(ENGINE_DET[(512, 896)])

        def paso(img):
            x, r, l, t, nh, nw = letterbox(img, 512, 896)  # un solo preprocesamiento
            mascara = post_segmentacion(seg(x), 512, 896, l, t, nh, nw, img.shape[:2])
            cajas = post_deteccion(det(x), r, l, t, img.shape[:2])
            return mascara, cajas
        return paso

    raise ValueError(config)


# ---------------------------------------------------------------------------
def correr(config, duracion, warmup):
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    print(f"== {config} == (hilos OpenCV: {cv2.getNumThreads()})")
    tegra = Tegrastats()
    resultado = {"config": config, "duracion_s": duracion}

    if config == "idle":
        time.sleep(5)
        tegra.iniciar()
        time.sleep(duracion)
        muestras = tegra.detener()
    else:
        imagenes = [cv2.imread(str(p)) for p in sorted(DIR_TEST.glob("*.jpg"))]
        paso = construir_pipeline(config)

        fin_warmup = time.time() + warmup
        i = 0
        while time.time() < fin_warmup:
            paso(imagenes[i % len(imagenes)])
            i += 1

        latencias = []
        n_frames = 0
        tegra.iniciar()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duracion:
            ts = time.perf_counter()
            paso(imagenes[n_frames % len(imagenes)])
            latencias.append((time.perf_counter() - ts) * 1000)
            n_frames += 1
        transcurrido = time.perf_counter() - t0
        muestras = tegra.detener()

        lat = np.array(latencias)
        resultado.update({
            "frames": n_frames,
            "fps": n_frames / transcurrido,
            "latencia_mediana_ms": float(np.median(lat)),
            "latencia_p99_ms": float(np.percentile(lat, 99)),
            "rss_proceso_mb": leer_rss_mb(),
        })

    tegra.cerrar()
    resultado["tegrastats"] = resumir_tegra(muestras)
    if "fps" in resultado and "vdd_in_mw" in resultado["tegrastats"]:
        resultado["energia_mj_por_frame"] = resultado["tegrastats"]["vdd_in_mw"]["media"] / resultado["fps"]

    with open(DIR_LOGS / f"{config}.json", "w") as f:
        json.dump(resultado, f, indent=2)

    tg = resultado["tegrastats"]
    print(f"   muestras tegrastats: {tg['n_muestras']}")
    if "fps" in resultado:
        print(f"   FPS {resultado['fps']:.1f} | latencia mediana {resultado['latencia_mediana_ms']:.2f} ms | "
              f"p99 {resultado['latencia_p99_ms']:.2f} ms | RSS {resultado['rss_proceso_mb']:.0f} MB")
    if "vdd_in_mw" in tg:
        print(f"   VDD_IN media {tg['vdd_in_mw']['media']:.0f} mW | RAM sistema max {tg['ram_mb']['max']:.0f} MB")
    print(f"   guardado en {DIR_LOGS / (config + '.json')}")


def resumen():
    datos = {}
    for c in CONFIGS:
        ruta = DIR_LOGS / f"{c}.json"
        if ruta.exists():
            datos[c] = json.load(open(ruta))
    if "idle" not in datos:
        print("Falta la medicion idle (linea base).")
        return

    p_idle = datos["idle"]["tegrastats"]["vdd_in_mw"]["media"]
    ram_idle = datos["idle"]["tegrastats"]["ram_mb"]["media"]
    print(f"Linea base idle: VDD_IN {p_idle:.0f} mW | RAM {ram_idle:.0f} MB\n")

    encabezado = (f"{'config':6s} {'FPS':>6s} {'lat_med':>8s} {'lat_p99':>8s} {'P_in_W':>7s} {'dP_W':>6s} "
                  f"{'mJ/frame':>9s} {'dRAM_MB':>8s} {'RSS_MB':>7s} {'GPU%':>5s} {'Tj_max':>7s}")
    print(encabezado)
    print("-" * len(encabezado))
    for c in CONFIGS[1:]:
        if c not in datos:
            continue
        d, tg = datos[c], datos[c]["tegrastats"]
        p = tg["vdd_in_mw"]["media"]
        print(f"{c:6s} {d['fps']:6.1f} {d['latencia_mediana_ms']:8.2f} {d['latencia_p99_ms']:8.2f} "
              f"{p / 1000:7.2f} {(p - p_idle) / 1000:6.2f} {d['energia_mj_por_frame']:9.1f} "
              f"{tg['ram_mb']['max'] - ram_idle:8.0f} {d['rss_proceso_mb']:7.0f} "
              f"{tg.get('gpu_pct', {}).get('media', float('nan')):5.0f} "
              f"{tg.get('temp_tj_c', {}).get('max', float('nan')):7.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--duracion", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--resumen", action="store_true")
    args = parser.parse_args()

    if args.resumen:
        resumen()
    elif args.config:
        correr(args.config, args.duracion, args.warmup)
    else:
        parser.print_help()
