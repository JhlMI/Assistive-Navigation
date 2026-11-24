"""Fine-tuning effect: segmentation models WITHOUT fine-tuning (Cityscapes weights) vs WITH
fine-tuning (La Paz), on the test set and on one harness image.

Same evaluation as scripts/eval_segmentacion_jetson.py:
  - ground truth: data/test_lapaz/test/*_mask.png, values remapped with _classes.csv
    (0 and 1 -> background, 2 building, 3 pole, 4 road, 5 sidewalk, 6 vegetation)
  - SegFormer: PIL bilinear resize to 896x512, /255, ImageNet normalization
  - YOLO26n-sem: letterbox 512x896 (gray 114), /255
  - confusion matrix accumulated over all images, at the original mask resolution
Fine-tuned models: the deployed ONNX files (same graphs as the TensorRT engines).
Cityscapes models: SegFormer-B0 from HuggingFace and yolo26n-sem.pt from Ultralytics (downloaded
once, needs internet); their 19 classes are mapped by NAME to ours:
  road->road, sidewalk->sidewalk, building/wall->building, pole->pole,
  vegetation/terrain->vegetation, everything else->background.

Usage (Jetson, yolo_env, app stopped):
    python scripts/finetune_compare.py
    python scripts/finetune_compare.py --image logs/navigation/snapshots/snap_20261104_160910_577.jpg
Output: logs/evidence/finetune/  results.csv, per_image.csv, comparison_<image>.png
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE = Path.home() / "Assistive-Navigation"
TEST_DIR = BASE / "data" / "test_lapaz" / "test"
ONNX_DIR = BASE / "modelos" / "segmentacion" / "onnx"
OUT_DIR = BASE / "logs" / "evidence" / "finetune"
H, W = 512, 896
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ["background", "building", "pole", "road", "sidewalk", "vegetation"]
SIDEWALK = CLASSES.index("sidewalk")
NAME_TO_OURS = {"road": 3, "sidewalk": 4, "building": 1, "wall": 1, "pole": 2,
                "vegetation": 5, "terrain": 5}
COLORS = np.array([[40, 40, 40], [70, 70, 160], [0, 220, 255], [128, 64, 128],
                   [60, 200, 60], [35, 120, 35]], np.uint8)   # BGR, same as evidence.py


# ---- ground truth --------------------------------------------------------------------
def load_remap(csv_path):
    """_classes.csv (Pixel Value, Class) -> lookup table raw value -> our class id."""
    lut = np.zeros(256, np.uint8)
    with open(csv_path) as f:
        next(f)
        for line in f:
            if "," not in line:
                continue
            v, name = [x.strip() for x in line.split(",", 1)]
            lut[int(v)] = CLASSES.index(name)
    return lut


# ---- preprocessing (identical to eval_segmentacion_jetson.py) ----------------------------
def pre_segformer(img_bgr):
    rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    x = np.asarray(rgb.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), None


def pre_yolo(img_bgr):
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


def to_class_map(out):
    out = np.asarray(out)
    if out.ndim == 4 and out.shape[1] > 1:          # per-class scores -> argmax
        out = out.argmax(1)
    return out.reshape(H, W)


def post(mask_512x896, pad, shape):
    m = mask_512x896.astype(np.uint8)
    if pad is not None:
        top, left, nh, nw = pad
        m = m[top:top + nh, left:left + nw]
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


# ---- models --------------------------------------------------------------------------
class OnnxSeg:
    def __init__(self, path, kind, lut=None):
        import onnxruntime as ort
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.s = ort.InferenceSession(str(path), providers=prov)
        self.kind, self.lut = kind, lut

    def __call__(self, img):
        x, pad = (pre_segformer if self.kind == "segformer" else pre_yolo)(img)
        m = post(to_class_map(self.s.run(None, {self.s.get_inputs()[0].name: x})[0]), pad, img.shape)
        return self.lut[m] if self.lut is not None else m


class HfSegformer:
    """SegFormer from HuggingFace: logits upsampled bilinearly to 512x896 + argmax (as in the ONNX)."""

    def __init__(self, name, lut):
        import torch
        from transformers import SegformerForSemanticSegmentation
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.m = SegformerForSemanticSegmentation.from_pretrained(name, use_safetensors=True)
        except Exception:
            # The Cityscapes checkpoint only ships pytorch_model.bin, and transformers refuses .bin
            # files with torch < 2.6. Load it ourselves with weights_only=True (official NVIDIA file).
            from huggingface_hub import hf_hub_download
            from transformers import SegformerConfig
            self.m = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(name))
            sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu", weights_only=True)
            missing, unexpected = self.m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
        self.m = self.m.to(self.dev).eval()
        self.lut = lut

    def __call__(self, img):
        x, _ = pre_segformer(img)
        with self.torch.no_grad():
            lg = self.m(pixel_values=self.torch.from_numpy(x).to(self.dev)).logits
            lg = self.torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)
            m = lg.argmax(1)[0].cpu().numpy()
        return self.lut[post(m, None, img.shape)]


def lut_from_names(names):
    lut = np.zeros(256, np.uint8)
    for i, n in names.items():
        lut[int(i)] = NAME_TO_OURS.get(str(n).strip().lower().replace(" ", ""), 0)
    return lut


def cityscapes_yolo_onnx():
    """Export yolo26n-sem.pt (Cityscapes) to ONNX once, same recipe as the deployed model."""
    out = ONNX_DIR / "yolo26n_sem_cityscapes_512x896.onnx"
    names_file = out.with_suffix(".names.txt")
    if not out.exists():
        import subprocess
        code = ("from ultralytics import YOLO; m = YOLO('yolo26n-sem.pt'); "
                "open('%s','w').write(repr(m.names)); "
                "p = m.export(format='onnx', imgsz=(%d, %d), opset=17, simplify=False); "
                "import shutil; shutil.move(p, '%s')" % (names_file, H, W, out))
        print("Exporting yolo26n-sem.pt (Cityscapes) to ONNX (first run only, downloads the weights)...")
        subprocess.run([sys.executable, "-c", code], check=True, cwd=str(ONNX_DIR))
    names = eval(names_file.read_text())
    return out, names


# ---- evaluation ----------------------------------------------------------------------
def iou_from_conf(conf):
    tp = np.diag(conf)
    return tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)


SIDEWALK_HIGHLIGHT = (255, 200, 0)          # BGR: bright cyan-blue, unmistakable against vegetation


def _font(size, bold=False):
    from PIL import ImageFont
    for f in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _text(img_bgr, items):
    """Draw UTF-8 text (accents) with PIL. items: (xy, text, size, bold, rgb, anchor)."""
    from PIL import ImageDraw
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    for xy, t, size, bold, rgb, anchor in items:
        d.text(xy, t, font=_font(size, bold), fill=rgb, anchor=anchor)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _fmt(v):
    return f"{v:.3f}".replace(".", ",")


def figure(img, preds, ious, path):
    """Column headers (without / with fine-tuning), row labels (model), sidewalk highlighted."""
    w = 900
    h = int(round(img.shape[0] * w / img.shape[1]))
    base = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    gap, lab_w, head_h, sub_h = 14, 70, 70, 46
    colors = COLORS.copy()
    colors[SIDEWALK] = SIDEWALK_HIGHLIGHT

    def panel(mask):
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(base, 0.5, colors[m], 0.5, 0)
        cnts, _ = cv2.findContours((m == SIDEWALK).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 3, cv2.LINE_AA)
        return out

    W_ = lab_w + 2 * w + gap
    top_h = int(round(h * 0.9))
    top_w = int(round(top_h * img.shape[1] / img.shape[0]))
    H_ = 20 + top_h + 30 + head_h + 2 * (h + sub_h) + gap + 70
    canvas = np.full((H_, W_, 3), 255, np.uint8)
    x0 = (W_ - top_w) // 2
    canvas[20:20 + top_h, x0:x0 + top_w] = cv2.resize(img, (top_w, top_h), interpolation=cv2.INTER_AREA)
    texts = [((x0, 20 + top_h + 6), "Imagen del arnés (OAK-D Lite, La Paz)", 22, False, (60, 60, 60), "la")]
    y = 20 + top_h + 30
    for c, lbl in enumerate(("Sin ajuste fino (pesos de Cityscapes)", "Con ajuste fino (dataset de La Paz)")):
        texts.append(((lab_w + c * (w + gap) + w // 2, y + head_h // 2), lbl, 28, True, (20, 20, 20), "mm"))
    y += head_h
    for r, model in enumerate(("SegFormer-B0", "YOLO26n-sem")):
        yy = y + r * (h + sub_h + gap)
        for c, tag in enumerate(("sin", "con")):
            xx = lab_w + c * (w + gap)
            m = preds[(model, tag)]
            canvas[yy:yy + h, xx:xx + w] = panel(m)
            sub = f"IoU de acera en el conjunto de prueba: {_fmt(ious[(model, tag)])}" if (model, tag) in ious else ""
            texts.append(((xx + w // 2, yy + h + sub_h // 2), sub, 21, (model, tag) in ious, (30, 30, 30), "mm"))
        # row label, written vertically
        lab = np.full((lab_w, h, 3), 255, np.uint8)
        lab = _text(lab, [((h // 2, lab_w // 2), model, 30, True, (20, 20, 20), "mm")])
        canvas[yy:yy + h, 0:lab_w] = cv2.rotate(lab, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # legend
    ly, x = H_ - 45, lab_w
    names = {"background": "fondo", "building": "edificio", "pole": "poste", "road": "calzada",
             "sidewalk": "acera", "vegetation": "vegetación"}
    for i, n in enumerate(CLASSES):
        cv2.rectangle(canvas, (x, ly - 14), (x + 28, ly + 14), colors[i].tolist(), -1)
        cv2.rectangle(canvas, (x, ly - 14), (x + 28, ly + 14), (90, 90, 90), 1)
        texts.append(((x + 38, ly), names[n], 22, n == "sidewalk", (30, 30, 30), "lm"))
        x += 70 + 13 * len(names[n])
    texts.append(((x + 20, ly), "contorno blanco = acera detectada", 22, False, (30, 30, 30), "lm"))
    cv2.imwrite(str(path), _text(canvas, texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(BASE / "logs/navigation/snapshots/snap_20261104_160910_577.jpg"))
    ap.add_argument("--no-test", action="store_true", help="only the figure (skip the test-set metrics)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    from transformers import AutoConfig
    cs_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    seg_cs = HfSegformer(cs_name, lut_from_names(AutoConfig.from_pretrained(cs_name).id2label))
    yolo_cs_path, yolo_names = cityscapes_yolo_onnx()
    print("YOLO26n-sem Cityscapes classes:", yolo_names)
    models = {
        ("SegFormer-B0", "sin"): seg_cs,
        ("SegFormer-B0", "con"): OnnxSeg(ONNX_DIR / "segformer_b0_512x896_mask.onnx", "segformer"),
        ("YOLO26n-sem", "sin"): OnnxSeg(yolo_cs_path, "yolo", lut_from_names(yolo_names)),
        ("YOLO26n-sem", "con"): OnnxSeg(ONNX_DIR / "yolo26n_sem_512x896.onnx", "yolo"),
    }

    ious = {}
    if args.no_test and (OUT_DIR / "results.csv").exists():
        with open(OUT_DIR / "results.csv") as f:
            for r in csv.DictReader(f):
                ious[(r["model"], r["fine_tuning"])] = float(r["iou_sidewalk"])
    if not args.no_test:
        lut_gt = load_remap(TEST_DIR / "_classes.csv")
        imgs = sorted(TEST_DIR.glob("*.jpg"))
        conf = {k: np.zeros((6, 6), np.int64) for k in models}
        per_image = []
        t0 = time.time()
        for n, p in enumerate(imgs, 1):
            img = cv2.imread(str(p))
            gt = lut_gt[np.array(Image.open(p.with_name(p.stem + "_mask.png")).convert("L"))]
            row = {"image": p.name, "gt_sidewalk_pct": round(100 * (gt == SIDEWALK).mean(), 2)}
            for k, model in models.items():
                pr = model(img)
                if pr.shape != gt.shape:
                    pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                conf[k] += np.bincount(gt.ravel().astype(np.int64) * 6 + pr.ravel(), minlength=36).reshape(6, 6)
                a, b = pr == SIDEWALK, gt == SIDEWALK
                row[f"{k[0]}_{k[1]}_iou_sidewalk"] = round((a & b).sum() / max((a | b).sum(), 1), 4)
            per_image.append(row)
            print(f"  {n}/{len(imgs)}  {time.time() - t0:5.0f} s", end="\r", flush=True)
        print()

        results = []
        print(f"\n{len(imgs)} test images\n")
        print(f"{'model':14s} {'fine-tune':10s} {'IoU sidewalk':>12s} {'mIoU 6 cl':>10s} {'sidewalk px %':>14s}")
        for k in models:
            iou = iou_from_conf(conf[k])
            ious[k] = iou[SIDEWALK]
            sw_pct = 100 * conf[k][:, SIDEWALK].sum() / conf[k].sum()
            results.append({"model": k[0], "fine_tuning": k[1], "iou_sidewalk": round(iou[SIDEWALK], 4),
                            "miou_6": round(iou.mean(), 4), "pred_sidewalk_pct": round(sw_pct, 2),
                            **{f"iou_{c}": round(v, 4) for c, v in zip(CLASSES, iou)}})
            print(f"{k[0]:14s} {k[1]:10s} {iou[SIDEWALK]:12.3f} {iou.mean():10.3f} {sw_pct:13.1f} %")
        gt_pct = 100 * conf[next(iter(models))][SIDEWALK, :].sum() / conf[next(iter(models))].sum()
        print(f"\nReal sidewalk in the test masks: {gt_pct:.1f} % of the pixels")
        for name, rows in (("results.csv", results), ("per_image.csv", per_image)):
            with open(OUT_DIR / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Could not read {args.image}")
    preds = {k: m(img) for k, m in models.items()}
    out = OUT_DIR / f"comparison_{Path(args.image).stem}.png"
    figure(img, preds, ious, out)
    print(f"\nSaved {out}\n      {OUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
