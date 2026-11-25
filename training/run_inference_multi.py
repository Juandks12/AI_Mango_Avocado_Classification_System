#!/usr/bin/env python3
# run_inference_multi.py
"""
Inference runner for trained object detectors saved under:
  runs/<dataset_name>/<model_name>/best.pth

Supports: fasterrcnn, ssdlite, effdet (if effdet installed).

Example:
  python run_inference_multi.py --data_root TODO3/aguacate.v1i.coco --model fasterrcnn --input test.jpg --conf 0.25
"""
import argparse
from pathlib import Path
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import time

# try optional effdet
try:
    from effdet import create_model as effdet_create_model
    HAS_EFFDET = True
except Exception:
    HAS_EFFDET = False

# ---------------- helpers ----------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def find_train_json(data_root: Path):
    # typical roboflow: data_root/train/_annotations.coco.json
    candidates = [
        data_root / "train",
        data_root
    ]
    for c in candidates:
        if c.exists():
            for f in c.iterdir():
                if f.suffix.lower() == ".json":
                    try:
                        j = json.loads(f.read_text(encoding="utf8"))
                        if "images" in j and "annotations" in j and "categories" in j:
                            return f
                    except Exception:
                        continue
    # fallback recursive
    for f in data_root.rglob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf8"))
            if "images" in j and "annotations" in j and "categories" in j:
                return f
        except Exception:
            continue
    return None

def build_class_names_from_coco(train_json: Path):
    coco = json.loads(train_json.read_text(encoding="utf8"))
    cats = coco.get("categories", [])
    if not cats:
        return []
    max_id = max([c["id"] for c in cats])
    names = [None] * (max_id + 1)
    for c in cats:
        cid = c["id"]
        names[cid] = c.get("name", f"class_{cid}")
    # fill missing
    for i in range(len(names)):
        if names[i] is None:
            names[i] = f"class_{i}"
    return names

def find_best_model_path(data_root: Path, model_name: str, explicit=None):
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        else:
            raise FileNotFoundError(f"Model path {explicit} not found")
    candidate = Path("runs") / data_root.name / model_name / "best.pth"
    if candidate.exists():
        return candidate
    # try other common names
    candidate2 = Path("runs") / data_root.name / model_name / "best_model.pth"
    if candidate2.exists():
        return candidate2
    # try epoch files and pick last
    ckdir = Path("runs") / data_root.name / model_name / "checkpoints"
    if ckdir.exists():
        files = sorted(list(ckdir.glob("epoch_*.pth")))
        if files:
            return files[-1]
    raise FileNotFoundError(f"No model found under runs/{data_root.name}/{model_name}/ (expected best.pth)")

def find_next_id(outdir: Path, prefix="pred_", pad=4):
    outdir.mkdir(parents=True, exist_ok=True)
    existing = list(outdir.glob(f"{prefix}*.json"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        try:
            n = int(p.stem.split("_")[-1])
            nums.append(n)
        except:
            pass
    return max(nums) + 1 if nums else 1

def draw_boxes(pil_img: Image.Image, boxes, labels, scores, class_names):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    for (x1,y1,x2,y2), lab, sc in zip(boxes, labels, scores):
        lab = int(lab)
        sc = float(sc)
        name = class_names[lab] if (0 <= lab < len(class_names)) else str(lab)
        text = f"{name} {sc:.2f}"
        # draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # text background
        try:
            tb = draw.textbbox((0,0), text, font=font)
            tw = tb[2] - tb[0]; th = tb[3] - tb[1]
        except Exception:
            tw, th = font.getsize(text)
        ty = y1 - th if y1 - th >= 0 else y1
        draw.rectangle([x1, ty, x1+tw, ty+th], fill="red")
        draw.text((x1, ty), text, fill="white", font=font)
    return pil_img

# ---------------- model builders ----------------
def build_fasterrcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def build_ssdlite(num_classes):
    # Prefer passing num_classes directly if torchvision supports it.
    try:
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return model
    except TypeError:
        # Fallback for older torchvision: attempt minimal adaptation.
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None)
        try:
            head = model.head
            # best-effort: if the classification head exposes a num_classes attribute, set it
            if hasattr(head.classification_head, 'num_classes'):
                head.classification_head.num_classes = num_classes
        except Exception:
            pass
        return model

def build_effdet(num_classes, effdet_name="tf_efficientdet_d1"):
    if not HAS_EFFDET:
        raise RuntimeError("EffDet not installed. pip install effdet timm")
    # bench_task='predict' to use DetBenchPredict internally if available
    model = effdet_create_model(effdet_name, bench_task='predict', num_classes=num_classes, pretrained=False)
    return model

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="e.g. TODO3/aguacate.v1i.coco")
    parser.add_argument("--model", type=str, required=True, choices=("fasterrcnn","ssdlite","effdet"))
    parser.add_argument("--model_path", type=str, default=None, help="path to .pth (optional)")
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--outdir", type=str, default="predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--effdet_name", type=str, default="tf_efficientdet_d1")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root}")

    train_json = find_train_json(data_root)
    if train_json is None:
        raise SystemExit("train JSON COCO not found inside data_root")
    class_names = build_class_names_from_coco(train_json)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[i] Device:", device)

    # find model file
    model_path = find_best_model_path(data_root, args.model, explicit=args.model_path)
    print("[i] Using model file:", model_path)

    # build model skeleton
    num_classes = len(class_names)
    if args.model == "fasterrcnn":
        model = build_fasterrcnn(num_classes)
    elif args.model == "ssdlite":
        model = build_ssdlite(num_classes)
    else:
        if not HAS_EFFDET:
            raise SystemExit("EffDet not available in this environment")
        model = build_effdet(num_classes, effdet_name=args.effdet_name)

    # capture model's configured input size (before loading weights)
    model_img_size = None
    try:
        conf = getattr(model, "config", None)
        if conf is not None:
            model_img_size = getattr(conf, "image_size", None)
    except Exception:
        model_img_size = None
    print("[i] model_img_size (from model.config):", model_img_size)

    # load weights (map to device)
    state = torch.load(str(model_path), map_location="cpu")
    # some checkpoints store a flat state dict whose keys are prefixed (e.g. 'model.*')
    if isinstance(state, dict):
        # detect nested wrappers like OrderedDict with 'model.' prefix
        keys = list(state.keys())
        if keys and all(k.startswith("model.") for k in keys[:10]):
            stripped = {k.replace("model.", "", 1): v for k, v in state.items()}
            state = stripped
    def _partial_load(state_dict, model):
        model_sd = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_sd:
                mv = model_sd[k]
                try:
                    if getattr(v, 'shape', None) == getattr(mv, 'shape', None):
                        filtered[k] = v
                    else:
                        skipped.append((k, getattr(v, 'shape', None), getattr(mv, 'shape', None)))
                except Exception:
                    skipped.append((k, None, getattr(mv, 'shape', None)))
            else:
                # key not present in model
                skipped.append((k, 'missing_in_model', None))
        try:
            model.load_state_dict(filtered, strict=False)
            print(f"[i] Partially loaded state dict: {len(filtered)} params; skipped {len(skipped)} mismatches")
            if skipped:
                for s in skipped[:10]:
                    print("  skipped:", s)
        except Exception as e:
            print("[!] Partial load failed:", e)

    try:
        model.load_state_dict(state)
        print("[i] State dict loaded (strict=True)")
    except Exception as e:
        print("[!] strict load failed:", e)
        try:
            model.load_state_dict(state, strict=False)
            print("[i] State dict loaded (strict=False)")
        except Exception as e2:
            print("[!] strict=False also failed:", e2)
            # attempt to only load parameters that match by shape
            if isinstance(state, dict):
                # if checkpoint wraps in {'state_dict': ...} try to extract
                if 'state_dict' in state and isinstance(state['state_dict'], dict):
                    cand = state['state_dict']
                else:
                    cand = state
                _partial_load(cand, model)

    model.to(device)
    model.eval()

    # load image
    inp = Path(args.input)
    if not inp.exists() or inp.suffix.lower() not in IMG_EXTS:
        raise SystemExit(f"Input image not found or invalid ext: {inp}")
    pil = Image.open(inp).convert("RGB")
    # transform to tensor [0,1]
    # For effdet backbones prefer the model's configured image size if available
    if args.model == "effdet":
        print(f"[d] model_img_size repr: {repr(model_img_size)} type: {type(model_img_size)}")
        print(f"[d] original pil.size: {pil.size}")
        img_size = model_img_size
        # Try to coerce model.config.image_size to two ints (handles list, tuple, OmegaConf ListConfig, etc.)
        if img_size is not None:
            try:
                # support indexable config objects
                new_w = int(img_size[0])
                new_h = int(img_size[1])
                if pil.size != (new_w, new_h):
                    print(f"[i] Resizing image to model config size: {(new_w, new_h)} (was {pil.size})")
                    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
            except Exception:
                # fallback: resize to multiples of 32 to help FPN alignment
                w, h = pil.size
                def _ceil_mul(x, m):
                    return int(((x + m - 1) // m) * m)
                new_w = _ceil_mul(w, 32)
                new_h = _ceil_mul(h, 32)
                if new_w != w or new_h != h:
                    print(f"[i] Resizing image to multiple-of-32: {(new_w, new_h)} (was {pil.size})")
                    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    print("[i] Input image size after preprocessing:", pil.size)
    img_tensor = torchvision.transforms.functional.to_tensor(pil).to(device)

    # inference
    with torch.no_grad():
        if args.model == "effdet":
            # effdet models expect a batched tensor (B,C,H,W), not a list like torchvision detectors
            out = model(img_tensor.unsqueeze(0))
            # normalize outputs: many effdet wrappers return a dict possibly with batched tensors
            if isinstance(out, dict):
                # typical case: dict with batched tensors (B,N,4) etc. -> take first batch
                for k in ("boxes", "labels", "scores"):
                    if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() >= 2:
                        if out[k].dim() > 1 and out[k].shape[0] > 0:
                            out[k] = out[k][0]
            elif isinstance(out, (list, tuple)):
                # try to find a dict inside the returned list/tuple
                found = None
                for e in out:
                    if isinstance(e, dict):
                        found = e
                        break
                if found is not None:
                    out = found
                    for k in ("boxes", "labels", "scores"):
                        if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() >= 2:
                            if out[k].dim() > 1 and out[k].shape[0] > 0:
                                out[k] = out[k][0]
                else:
                    # no dict found: print shapes for debugging and attempt safe fallback
                    print("[w] effdet returned list/tuple without dict. Elements:")
                    for i, e in enumerate(out):
                        try:
                            if isinstance(e, torch.Tensor):
                                print(f"  - element[{i}] tensor shape: {tuple(e.shape)} dtype={e.dtype}")
                            else:
                                print(f"  - element[{i}] type: {type(e)} repr: {repr(e)[:200]}")
                        except Exception:
                            print(f"  - element[{i}] repr failed")
                    # safe fallback: empty results
                    print("[e] Cannot decode effdet output format — returning zero detections.")
                    out = {"boxes": torch.empty((0,4), device=device), "labels": torch.empty((0,), device=device), "scores": torch.empty((0,), device=device)}
            else:
                # single tensor or unexpected type
                if isinstance(out, torch.Tensor):
                    print(f"[w] effdet returned a Tensor of shape {tuple(out.shape)}; attempting to decode.")
                    t = out
                    try:
                        t = t.to(device)
                    except Exception:
                        pass
                    # common effdet output format observed: (B, N, 6) where columns are [x1,y1,x2,y2,score,label]
                    if t.dim() == 3:
                        B, N, C = t.shape
                        if B >= 1 and C >= 5:
                            t0 = t[0]
                            boxes_t = t0[:, :4]
                            scores_t = t0[:, 4]
                            if C >= 6:
                                labels_t = t0[:, 5].long()
                            else:
                                labels_t = torch.zeros((N,), dtype=torch.long, device=device)
                            out = {"boxes": boxes_t, "labels": labels_t, "scores": scores_t}
                        else:
                            print("[e] Tensor shape not decodable (need last-dim >=5). Returning zero detections.")
                            out = {"boxes": torch.empty((0,4), device=device), "labels": torch.empty((0,), device=device), "scores": torch.empty((0,), device=device)}
                    elif t.dim() == 2:
                        # maybe (N,6)
                        N, C = t.shape
                        if C >= 5:
                            t0 = t
                            boxes_t = t0[:, :4]
                            scores_t = t0[:, 4]
                            if C >= 6:
                                labels_t = t0[:, 5].long()
                            else:
                                labels_t = torch.zeros((N,), dtype=torch.long, device=device)
                            out = {"boxes": boxes_t, "labels": labels_t, "scores": scores_t}
                        else:
                            print("[e] 2D tensor last-dim too small to decode. Returning zero detections.")
                            out = {"boxes": torch.empty((0,4), device=device), "labels": torch.empty((0,), device=device), "scores": torch.empty((0,), device=device)}
                    else:
                        print(f"[w] effdet returned unexpected tensor ndim={t.dim()}; expected 2 or 3. Returning zero detections.")
                        out = {"boxes": torch.empty((0,4), device=device), "labels": torch.empty((0,), device=device), "scores": torch.empty((0,), device=device)}
                else:
                    print(f"[w] effdet returned unexpected type {type(out)}; expected dict or tensor.\n[e] Returning zero detections.")
                    out = {"boxes": torch.empty((0,4), device=device), "labels": torch.empty((0,), device=device), "scores": torch.empty((0,), device=device)}
        else:
            out = model([img_tensor])[0]

    boxes = out.get("boxes", torch.empty((0,4), device=device)).cpu().numpy()
    labels = out.get("labels", torch.empty((0,), device=device)).cpu().numpy().astype(int)
    scores = out.get("scores", torch.empty((0,), device=device)).cpu().numpy()

    # filter by conf
    keep = [i for i,s in enumerate(scores) if s >= args.conf]
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # prepare output folder
    outdir = Path(args.outdir) / f"{data_root.name}_{args.model}"
    outdir.mkdir(parents=True, exist_ok=True)

    nid = find_next_id(outdir, prefix="pred_", pad=4)
    id_str = str(nid).zfill(4)
    img_out = outdir / f"pred_{id_str}.jpg"
    json_out = outdir / f"pred_{id_str}.json"

    # draw
    pil_annot = pil.copy()
    pil_annot = draw_boxes(pil_annot, boxes.tolist(), labels.tolist(), scores.tolist(), class_names)
    pil_annot.save(img_out)
    print("[i] Annotated saved:", img_out)

    # build json
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    preds = []
    for i, (b, l, s) in enumerate(zip(boxes.tolist(), labels.tolist(), scores.tolist())):
        x1,y1,x2,y2 = b
        w = x2 - x1; h = y2 - y1
        preds.append({
            "id": i+1,
            "category_id": int(l),
            "category_name": class_names[int(l)] if int(l) < len(class_names) else str(int(l)),
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(s)
        })

    meta = {
        "input": str(inp),
        "model_path": str(model_path),
        "dataset": data_root.name,
        "model": args.model,
        "timestamp_utc": ts,
        "num_detections": len(preds),
        "predictions": preds
    }

    with open(json_out, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[i] JSON saved:", json_out)
    print("[i] Done.")

if __name__ == "__main__":
    main()
