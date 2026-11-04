"""
Linea base sin exportar: modelos PyTorch FP32 (tal como salieron del entrenamiento)
corriendo en la GPU del Jetson Orin Nano.

Mide, con el mismo alcance que la evaluacion de TensorRT/ONNX:
  - Precision en las 56 imagenes de test (control: debe coincidir con Colab)
  - Tiempo por etapa: preprocesamiento, inferencia (copia a GPU + modelo + upsampling +
    argmax + copia de la mascara a CPU) y postprocesamiento
  - Solo modelo: N iteraciones sobre un tensor fijo en GPU (equivalente a trtexec)
  - Memoria de GPU del proceso (memoria unificada en el Jetson)

Los tres modelos entregan la misma salida que sus versiones exportadas: mascara uint8 (1, 512, 896).

Uso (un modelo por proceso, para que la memoria no se contamine):
    python eval_pytorch_raw_jetson.py --modelo segformer_b0
    python eval_pytorch_raw_jetson.py --modelo segformer_b1
    python eval_pytorch_raw_jetson.py --modelo yolo26n_sem
    python eval_pytorch_raw_jetson.py --resumen
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HILOS_OPENCV = cv2.getNumThreads()  # se guarda antes de importar ultralytics (que lo baja a 1)

H, W = 512, 896
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE = Path.home() / "Assistive-Navigation"
DIR_TEST = BASE / "data" / "test_lapaz" / "test"
DIR_PYTORCH = BASE / "modelos" / "pytorch"
DIR_LOGS = BASE / "logs" / "benchmark_pytorch"
MODELOS = {"segformer_b0": "segformer", "segformer_b1": "segformer", "yolo26n_sem": "yolo"}


# ---------------------------------------------------------------------------
# Clases y metricas (identicas a eval_segmentacion_jetson.py)
# ---------------------------------------------------------------------------
def cargar_clases(ruta_csv):
    clases_por_valor = {}
    with open(ruta_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for fila in reader:
            clases_por_valor[int(fila[0].strip())] = fila[1].strip()
    nombres = []
    for n in clases_por_valor.values():
        if n not in nombres:
            nombres.append(n)
    tabla = np.zeros(256, dtype=np.uint8)
    for valor, n in clases_por_valor.items():
        tabla[valor] = nombres.index(n)
    return nombres, tabla


def metricas(conf, nombres):
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    i_side, i_road, i_bg = nombres.index("sidewalk"), nombres.index("road"), nombres.index("background")
    tp = conf[i_side, i_side]
    fp = conf[:, i_side].sum() - tp
    fn = conf[i_side].sum() - tp
    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    return {
        "miou": float(iou.mean()),
        "iou_por_clase": {n: float(iou[i]) for i, n in enumerate(nombres)},
        "precision_transitable": float(tp / max(tp + fp, 1)),
        "recall_transitable": float(tp / max(tp + fn, 1)),
        "calle_como_acera": float(norm[i_road, i_side]),
        "background_como_acera": float(norm[i_bg, i_side]),
    }


def resumen_tiempos(lista_ms):
    a = np.array(lista_ms)
    return {"mediana_ms": float(np.median(a)), "media_ms": float(a.mean()),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "min_ms": float(a.min()), "max_ms": float(a.max())}


# ---------------------------------------------------------------------------
# Pre y postprocesamiento (identicos a eval_segmentacion_jetson.py)
# ---------------------------------------------------------------------------
def preprocesar(img_bgr, tipo):
    if tipo == "segformer":
        rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        x = np.asarray(rgb.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), None

    h0, w0 = img_bgr.shape[:2]
    r = min(H / h0, W / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    dw, dh = (W - nw) / 2, (H - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    x = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (top, left, nh, nw)


def postprocesar(mascara, tipo, pad, forma_original):
    m = np.asarray(mascara).reshape(H, W).astype(np.uint8)
    if tipo == "yolo":
        top, left, nh, nw = pad
        m = m[top:top + nh, left:left + nw]
    alto, ancho = forma_original
    return cv2.resize(m, (ancho, alto), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Modelos PyTorch FP32
# ---------------------------------------------------------------------------
def cargar_modelo(nombre, torch, F, n_clases):
    info = {}
    if MODELOS[nombre] == "segformer":
        import transformers
        from transformers import SegformerForSemanticSegmentation

        modelo = SegformerForSemanticSegmentation.from_pretrained(DIR_PYTORCH / nombre).eval().cuda().float()
        info["transformers"] = transformers.__version__

        def forward(x):
            logits = modelo(pixel_values=x).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            return logits.argmax(dim=1).to(torch.uint8)

        n_params = sum(p.numel() for p in modelo.parameters())
        return forward, n_params, info

    import ultralytics
    from ultralytics import YOLO

    cv2.setNumThreads(HILOS_OPENCV)  # ultralytics lo bajo a 1 al importarse
    yolo = YOLO(str(DIR_PYTORCH / nombre / "best.pt"))
    red = yolo.model
    try:
        red = red.fuse()  # la exportacion tambien fusiona Conv+BN; asi la comparacion es justa
        info["fusionado"] = True
    except Exception as e:
        info["fusionado"] = f"no ({e})"
    red = red.eval().cuda().float()
    info["ultralytics"] = ultralytics.__version__
    info["hilos_opencv_restaurados"] = cv2.getNumThreads()

    def forward(x):
        salida = red(x)
        if isinstance(salida, (list, tuple)):
            salida = salida[0]
        if salida.dim() == 4:  # logits (1, C, h, w)
            if salida.shape[1] != n_clases:
                raise RuntimeError(f"Salida inesperada de YOLO26n-sem: {tuple(salida.shape)}")
            if tuple(salida.shape[-2:]) != (H, W):
                salida = F.interpolate(salida, size=(H, W), mode="bilinear", align_corners=False)
            return salida.argmax(dim=1).to(torch.uint8)
        return salida.to(torch.uint8)  # ya es mascara de clases

    n_params = sum(p.numel() for p in red.parameters())
    return forward, n_params, info


# ---------------------------------------------------------------------------
def evaluar(nombre, rondas, iteraciones, warmup):
    import torch
    import torch.nn.functional as F

    tipo = MODELOS[nombre]
    nombres, tabla_remapeo = cargar_clases(DIR_TEST / "_classes.csv")
    n_clases = len(nombres)
    imagenes = sorted(DIR_TEST.glob("*.jpg"))
    print(f"== {nombre} (PyTorch FP32, sin exportar) ==")
    print(f"   torch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}")

    torch.cuda.reset_peak_memory_stats()
    mem_inicio = torch.cuda.memory_allocated()
    forward, n_params, info = cargar_modelo(nombre, torch, F, n_clases)
    mem_modelo_mb = (torch.cuda.memory_allocated() - mem_inicio) / 2 ** 20
    print(f"   parametros {n_params / 1e6:.2f} M | memoria de pesos {mem_modelo_mb:.0f} MB | {info}")

    datos = [(ruta, cv2.imread(str(ruta))) for ruta in imagenes]

    with torch.inference_mode():
        # Forma de la salida (control) y calentamiento
        x0, _ = preprocesar(datos[0][1], tipo)
        x0_gpu = torch.from_numpy(x0).cuda()
        print(f"   salida del modelo: {tuple(forward(x0_gpu).shape)} (esperado (1, {H}, {W}))")
        for _ in range(warmup):
            forward(x0_gpu)
        torch.cuda.synchronize()

        # 1) Por imagen: mismo alcance que la evaluacion de TensorRT
        conf = np.zeros((n_clases, n_clases), dtype=np.int64)
        t_pre, t_inf, t_post, t_total = [], [], [], []
        for ronda in range(rondas):
            for ruta, img in datos:
                t0 = time.perf_counter()
                x, pad = preprocesar(img, tipo)
                t1 = time.perf_counter()
                mascara = forward(torch.from_numpy(x).cuda()).cpu().numpy()  # .cpu() sincroniza
                t2 = time.perf_counter()
                pred = postprocesar(mascara, tipo, pad, img.shape[:2]).astype(np.int64)
                t3 = time.perf_counter()

                t_pre.append((t1 - t0) * 1000)
                t_inf.append((t2 - t1) * 1000)
                t_post.append((t3 - t2) * 1000)
                t_total.append((t3 - t0) * 1000)

                if ronda == 0:
                    gt_raw = np.array(Image.open(ruta.with_name(ruta.stem + "_mask.png")).convert("L"))
                    gt = tabla_remapeo[gt_raw].astype(np.int64)
                    idx = gt.flatten() * n_clases + pred.flatten()
                    conf += np.bincount(idx, minlength=n_clases ** 2).reshape(n_clases, n_clases)

        # 2) Solo modelo: tensor fijo en GPU, equivalente a trtexec
        lat_modelo = []
        inicio = time.perf_counter()
        for _ in range(iteraciones):
            ts = time.perf_counter()
            forward(x0_gpu)
            torch.cuda.synchronize()
            lat_modelo.append((time.perf_counter() - ts) * 1000)
        duracion = time.perf_counter() - inicio

    resultado = metricas(conf, nombres)
    resultado.update({
        "modelo": nombre,
        "backend": "PyTorch FP32 (sin exportar)",
        "parametros_M": round(n_params / 1e6, 3),
        "info": info,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "rondas_por_imagen": rondas,
        "tiempos": {
            "preprocesamiento": resumen_tiempos(t_pre),
            "inferencia": resumen_tiempos(t_inf),
            "postprocesamiento": resumen_tiempos(t_post),
            "total": resumen_tiempos(t_total),
        },
        "solo_modelo": {**resumen_tiempos(lat_modelo), "iteraciones": iteraciones,
                        "fps": iteraciones / duracion},
        "memoria_gpu": {
            "pesos_mb": mem_modelo_mb,
            "pico_asignado_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "pico_reservado_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
            "nota": "memoria unificada CPU/GPU del Jetson; solo cuenta lo asignado por PyTorch",
        },
    })

    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    with open(DIR_LOGS / f"{nombre}_pytorch.json", "w") as f:
        json.dump(resultado, f, indent=2)

    tt, sm, mem = resultado["tiempos"], resultado["solo_modelo"], resultado["memoria_gpu"]
    print(f"   mIoU {resultado['miou']:.4f} | sidewalk {resultado['iou_por_clase']['sidewalk']:.4f} | "
          f"precision transitable {resultado['precision_transitable']:.4f}")
    print(f"   por imagen (mediana ms): pre {tt['preprocesamiento']['mediana_ms']:.2f} | "
          f"inferencia {tt['inferencia']['mediana_ms']:.2f} | post {tt['postprocesamiento']['mediana_ms']:.2f} | "
          f"total {tt['total']['mediana_ms']:.2f}")
    print(f"   solo modelo: mediana {sm['mediana_ms']:.2f} ms | p99 {sm['p99_ms']:.2f} ms | {sm['fps']:.1f} FPS")
    print(f"   memoria GPU: pesos {mem['pesos_mb']:.0f} MB | pico asignado {mem['pico_asignado_mb']:.0f} MB | "
          f"pico reservado {mem['pico_reservado_mb']:.0f} MB")
    print(f"   guardado en {DIR_LOGS / (nombre + '_pytorch.json')}\n")


def resumen():
    filas = []
    for nombre in MODELOS:
        ruta = DIR_LOGS / f"{nombre}_pytorch.json"
        if not ruta.exists():
            print(f"[falta] {ruta}")
            continue
        d = json.load(open(ruta))
        filas.append((nombre, d))

    print(f"{'modelo':14s} {'mIoU':>7s} {'inf_med':>8s} {'total_med':>9s} {'solo_med':>9s} {'solo_p99':>9s} "
          f"{'FPS':>6s} {'mem_pico':>9s}")
    for nombre, d in filas:
        print(f"{nombre:14s} {d['miou']:7.4f} {d['tiempos']['inferencia']['mediana_ms']:8.2f} "
              f"{d['tiempos']['total']['mediana_ms']:9.2f} {d['solo_modelo']['mediana_ms']:9.2f} "
              f"{d['solo_modelo']['p99_ms']:9.2f} {d['solo_modelo']['fps']:6.1f} "
              f"{d['memoria_gpu']['pico_asignado_mb']:9.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=list(MODELOS))
    parser.add_argument("--rondas", type=int, default=3, help="Pasadas por las 56 imagenes para los tiempos")
    parser.add_argument("--iteraciones", type=int, default=500, help="Iteraciones de la medicion solo-modelo")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--resumen", action="store_true")
    args = parser.parse_args()

    if args.resumen:
        resumen()
    elif args.modelo:
        evaluar(args.modelo, args.rondas, args.iteraciones, args.warmup)
    else:
        parser.print_help()
