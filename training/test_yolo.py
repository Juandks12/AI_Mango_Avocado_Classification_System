# test_yolo_save.py
from ultralytics import YOLO
import cv2
import argparse
import os
import json
from pathlib import Path
from datetime import datetime

def infer_dataset_label(weights_path: str, fallback: str = "unknown"):
    wp = weights_path.lower()
    if "aguacate" in wp:
        return "aguacate"
    if "mango" in wp:
        return "mango"
    # try other heuristics: parent folder name
    p = Path(weights_path)
    for part in p.parts[::-1]:
        if "aguacate" in part.lower():
            return "aguacate"
        if "mango" in part.lower():
            return "mango"
    return fallback

def find_next_id(outdir: Path, prefix="pred_", pad=4):
    outdir.mkdir(parents=True, exist_ok=True)
    existing = list(outdir.glob(f"{prefix}*.json"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[-1]))
        except:
            pass
    return max(nums) + 1 if nums else 1

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [float(x1), float(y1), float(w), float(h)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True,
                        help="Ruta al modelo YOLOv8 (*.pt)")
    parser.add_argument("--img", type=str, required=True,
                        help="Ruta a la imagen a evaluar")
    parser.add_argument("--save", action="store_true",
                        help="Guardar imagen con bounding boxes y JSON")
    parser.add_argument("--out_root", type=str, default="detections",
                        help="Carpeta raíz donde guardar las detecciones")
    parser.add_argument("--dataset", type=str, default=None,
                        help="(opcional) nombre del dataset: 'aguacate' o 'mango' (si no se puede inferir)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold to filter detections")
    args = parser.parse_args()

    # -----------------------------
    # Validaciones básicas
    # -----------------------------
    if not os.path.exists(args.weights):
        print(f"[ERROR] Modelo no encontrado: {args.weights}")
        return

    if not os.path.exists(args.img):
        print(f"[ERROR] Imagen no encontrada: {args.img}")
        return

    print("\n===== Probando modelo YOLOv8 =====")
    print(f"Modelo: {args.weights}")
    print(f"Imagen: {args.img}")
    print(f"Threshold: {args.conf}")

    # -----------------------------
    # Cargar el modelo
    # -----------------------------
    model = YOLO(args.weights)

    # -----------------------------
    # Inferencia
    # -----------------------------
    results = model(args.img, conf=args.conf)  # pass conf here to early filter
    result = results[0]

    # Collect detections
    boxes_xyxy = []
    boxes_xywh = []
    labels = []
    scores = []

    # result.boxes may be a Boxes object; iterate robustly
    for b in result.boxes:
        # b.xyxy, b.cls, b.conf might be tensors or arrays depending on version
        try:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
        except Exception:
            # fallback if direct access differs
            try:
                xyxy = b.xyxy.tolist()[0]
            except Exception:
                # try result.boxes.xyxy if above fails
                xyxy = None

        try:
            cls_idx = int(b.cls[0].cpu().numpy().item())
        except Exception:
            try:
                cls_idx = int(b.cls[0].item())
            except Exception:
                cls_idx = int(b.cls)

        try:
            conf_val = float(b.conf[0].cpu().numpy().item())
        except Exception:
            try:
                conf_val = float(b.conf[0].item())
            except Exception:
                conf_val = float(b.conf)

        if xyxy is None:
            # try another way: result.boxes.xyxy (Nx4 tensor/array)
            try:
                all_xyxy = result.boxes.xyxy.cpu().numpy().tolist()
                all_cls = result.boxes.cls.cpu().numpy().astype(int).tolist()
                all_conf = result.boxes.conf.cpu().numpy().tolist()
                # replace lists and break (we will fill from these)
                boxes_xyxy = all_xyxy
                labels = all_cls
                scores = all_conf
                break
            except Exception:
                continue

        boxes_xyxy.append([float(x) for x in xyxy])
        boxes_xywh.append(xyxy_to_xywh(xyxy))
        labels.append(cls_idx)
        scores.append(float(conf_val))

    # If boxes_xyxy is empty but result.boxes has attributes, try extracting directly
    if not boxes_xyxy:
        try:
            bx = result.boxes.xyxy.cpu().numpy().tolist()
            labs = result.boxes.cls.cpu().numpy().astype(int).tolist()
            scr = result.boxes.conf.cpu().numpy().tolist()
            boxes_xyxy = bx
            labels = labs
            scores = scr
            boxes_xywh = [xyxy_to_xywh(b) for b in boxes_xyxy]
        except Exception:
            pass

    # Print detections
    print("\n===== DETECCIONES =====")
    names_map = result.names if hasattr(result, "names") else {}
    for i, (b, l, s) in enumerate(zip(boxes_xyxy, labels, scores), start=1):
        label_name = names_map[int(l)] if int(l) in names_map else str(l)
        print(f"{i}. {label_name} -> {s:.2f}  {b}")

    # -----------------------------
    # Guardar resultado visual y JSON
    # -----------------------------
    if args.save:
        out_root = Path(args.out_root)
        # infer dataset label if not provided
        dataset_label = args.dataset if args.dataset else infer_dataset_label(args.weights, fallback="unknown")
        model_name = Path(args.weights).stem  # e.g. best or aguacate_yolov8m_weights or similar
        # build out dir: detections/<dataset_label>/<model_name>/
        out_dir = out_root / dataset_label / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # determine next id
        next_id = find_next_id(out_dir, prefix="pred_", pad=4)
        id_str = str(next_id).zfill(4)
        img_out_path = out_dir / f"pred_{id_str}.jpg"
        json_out_path = out_dir / f"pred_{id_str}.json"

        # draw boxes on image (use cv2 for saving to keep color order)
        img_bgr = cv2.imread(args.img)
        if img_bgr is None:
            # fallback via PIL -> cv2
            from PIL import Image
            pil_img = Image.open(args.img).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # annotate
        for (x1, y1, x2, y2), lab, sc in zip(boxes_xyxy, labels, scores):
            x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
            label_name = names_map[int(lab)] if int(lab) in names_map else str(lab)
            text = f"{label_name} {sc:.2f}"
            # put label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            ty = y1i - th - 4 if y1i - th - 4 > 0 else y1i + th + 4
            cv2.rectangle(img_bgr, (x1i, ty - th - 2), (x1i + tw, ty + 2), (0, 0, 255), -1)
            cv2.putText(img_bgr, text, (x1i, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # save annotated image
        cv2.imwrite(str(img_out_path), img_bgr)
        print(f"\n[i] Imagen anotada guardada en: {img_out_path}")

        # build json metadata
        ts = datetime.utcnow().isoformat() + "Z"
        preds = []
        for i, (b, lab, sc) in enumerate(zip(boxes_xyxy, labels, scores), start=1):
            x1, y1, x2, y2 = b
            w = x2 - x1
            h = y2 - y1
            category_id = int(lab)
            category_name = names_map[category_id] if category_id in names_map else str(category_id)
            preds.append({
                "id": i,
                "image_id": next_id,
                "category_id": category_id,
                "category_name": category_name,
                "bbox": [float(x1), float(y1), float(w), float(h)],  # x,y,w,h
                "score": float(sc)
            })
        meta = {
            "input_path": str(Path(args.img).resolve()),
            "model_path": str(Path(args.weights).resolve()),
            "dataset": dataset_label,
            "model_name": model_name,
            "timestamp_utc": ts,
            "num_detections": len(preds),
            "predictions": preds
        }
        with open(json_out_path, "w", encoding="utf8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[i] JSON de predicción guardado en: {json_out_path}")

    print("\nListo.\n")

if __name__ == "__main__":
    main()
