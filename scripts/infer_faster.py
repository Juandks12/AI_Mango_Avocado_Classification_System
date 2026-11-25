#!/usr/bin/env python3
"""
infer_faster.py — versión con soporte para mango y aguacate.

Reglas:
- Cada modelo tiene 2 clases (1=defecto, 2=óptimo), más background.
- Si no se envía --class_names, detecta automáticamente según el nombre del modelo:
      - si contiene "aguacate": ["aguacate_defecto", "aguacate_optimo"]
      - si contiene "mango":   ["mango_defecto", "mango_optimo"]
"""

import json
from pathlib import Path
from datetime import datetime
import argparse
import traceback

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------
#   Construcción del modelo
# --------------------------------------------------------
def build_faster(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --------------------------------------------------------
#   Detectar número de clases desde el state_dict
# --------------------------------------------------------
def infer_num_classes_from_state(state_dict):
    possible = [
        "roi_heads.box_predictor.cls_score.weight",
        "model.roi_heads.box_predictor.cls_score.weight",
        "module.roi_heads.box_predictor.cls_score.weight",
    ]
    for k in possible:
        if k in state_dict:
            return int(state_dict[k].shape[0])
    return None


# --------------------------------------------------------
#   Cargar checkpoint robustamente
# --------------------------------------------------------
def load_checkpoint(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):

        if "model" in data and isinstance(data["model"], dict):
            return data["model"]

        if "state_dict" in data and isinstance(data["state_dict"], dict):
            return data["state_dict"]

        if all(hasattr(v, "shape") for v in list(data.values())[:5]):
            return data

    raise RuntimeError("Formato de checkpoint no reconocido.")


# --------------------------------------------------------
#   Mapeo de labels COCO → nombres
# --------------------------------------------------------
def map_label_to_name(label, class_names):
    """Map a model label to a human name.

    Accept either:
    - class_names without a background entry (e.g. ["mango_defecto", "mango_optimo"]) -> label 1 maps to index 0
    - class_names with background at index 0 (e.g. ["background","mango_defecto","mango_optimo"]) -> label 1 maps to index 1

    This makes the function robust against callers that include the background token
    (the Flask app's `MODEL_CLASS_NAMES` currently includes background at index 0).
    """
    if not class_names:
        return f"class_{label}"

    # If caller provided a background token at index 0, use it directly: label==0->background, label==1->class_names[1]
    try:
        if len(class_names) >= 1 and str(class_names[0]).lower() == "background":
            # label is integer produced by torchvision: background is usually 0
            if 0 <= label < len(class_names):
                return class_names[int(label)]
            return f"class_{label}"
    except Exception:
        pass

    # Otherwise assume class_names list corresponds to labels starting at 1
    if 1 <= label <= len(class_names):
        return class_names[int(label) - 1]
    return f"class_{label}"


# --------------------------------------------------------
#   Dibujar cajas
# --------------------------------------------------------
def draw_boxes(img, preds):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for p in preds:
        x, y, w, h = p["bbox"]
        x2, y2 = x + w, y + h
        txt = f"{p['category_name']} {p['score']:.2f}"

        draw.rectangle([x, y, x2, y2], outline="red", width=3)
        draw.text((x, y - 15 if y >= 15 else y + 5), txt, fill="red", font=font)

    return img


# --------------------------------------------------------
#   Auto-selección de clases según nombre del checkpoint
# --------------------------------------------------------
def auto_class_names(model_path: Path):
    name = model_path.name.lower()

    # ⭐⭐ CORRECCIÓN APLICADA AQUÍ: Invertimos el orden para alinear 
    # la predicción del modelo (Label 1) con la etiqueta correcta ('defecto'). ⭐⭐
    if "aguacate" in name:
        return ["aguacate_defecto", "aguacate_optimo"]

    if "mango" in name:
        return ["mango_defecto", "mango_optimo"]

    print("[WARN] No se pudo detectar fruta a partir del nombre del modelo.")
    return None


# --------------------------------------------------------
#   INFERENCIA PRINCIPAL
# --------------------------------------------------------
def infer_faster(model_path, img_path, out_img_path, out_json_path,
                     conf=0.25, class_names=None, device=None):

    model_path = Path(model_path)
    img_path = Path(img_path)
    out_img_path = Path(out_img_path)
    out_json_path = Path(out_json_path)

    # AUTO-DETECTAR FRUTA
    if class_names is None:
        class_names = auto_class_names(model_path)
        if class_names:
            print("[i] Usando clases detectadas automáticamente:", class_names)

    # Cargar checkpoint
    state = load_checkpoint(model_path)
    num_classes = infer_num_classes_from_state(state)

    if num_classes is None:
        num_classes = 3   # bg + 2 clases
        print("[WARN] No se detectó num_classes. Usando 3.")

    print(f"[i] num_classes = {num_classes}")

    # Construir modelo
    model = build_faster(num_classes)
    model.load_state_dict(state, strict=False)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    # Cargar imagen
    im = Image.open(img_path).convert("RGB")
    tensor = TF.to_tensor(im).to(device)

    with torch.no_grad():
        out = model([tensor])[0]

    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()

    preds = []
    id_count = 1

    for b, l, s in zip(boxes, labels, scores):
        if s < conf:
            continue
        x1, y1, x2, y2 = b
        preds.append({
            "id": id_count,
            "category_id": int(l),
            "category_name": map_label_to_name(int(l), class_names),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(s)
        })
        id_count += 1

    # Dibujar y guardar imagen
    out_img = draw_boxes(im.copy(), preds)
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_img_path)

    # Guardar JSON
    meta = {
        "input": str(img_path),
        "model": str(model_path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_detecciones": len(preds),
        "predictions": preds
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[i] Inferencia completada.")

    return meta


# --------------------------------------------------------
# CLI
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_img", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--class_names", type=str, default=None)

    args = parser.parse_args()

    class_names = None
    if args.class_names:
        class_names = [c.strip() for c in args.class_names.split(",")]

    infer_faster(
        model_path=args.model_path,
        img_path=args.input,
        out_img_path=args.out_img,
        out_json_path=args.out_json,
        conf=args.conf,
        class_names=class_names
    )