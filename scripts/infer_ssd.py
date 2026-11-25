# scripts/infer_ssd.py
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
from datetime import datetime

def build_ssdlite(num_classes=3):
    # Prefer using the public constructor that accepts `num_classes` where available.
    try:
        # Newer torchvision versions accept num_classes directly
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return model
    except TypeError:
        # Fallback for older torchvision: create default and try to adapt classification head
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=None)
        try:
            import torch.nn as nn
            head = model.head
            # try multiple possible attribute layouts
            if hasattr(head, "classification_head"):
                ch = head.classification_head
                # common layout: ch.conv is a Sequential or ModuleList
                if hasattr(ch, "conv") and len(ch.conv) > 0:
                    in_ch = ch.conv[0].in_channels
                    num_anchors = getattr(ch, "num_anchors", None)
                    if num_anchors is None and hasattr(ch, "classification_head"):
                        num_anchors = getattr(ch.classification_head, "num_anchors", None)
                    if num_anchors is None:
                        num_anchors = 6
                    new_out = num_anchors * num_classes
                    # replace last conv
                    ch.conv[-1] = nn.Conv2d(in_ch, new_out, kernel_size=3, padding=1)
                    ch.num_classes = num_classes
            return model
        except Exception:
            # As a last resort return the model as-is (inference may still work if state dict matches)
            return model

def infer_ssd(model_path, img_path, out_img_path, out_json_path, conf=0.85, class_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ssdlite(3)
    # Load checkpoint robustly: accept dicts with 'model' or 'state_dict' keys,
    # strip possible 'module.' prefixes and copy only matching-shape params.
    raw = torch.load(model_path, map_location="cpu")
    if isinstance(raw, dict):
        if "model" in raw:
            ck = raw["model"]
        elif "state_dict" in raw:
            ck = raw["state_dict"]
        else:
            ck = raw
    else:
        ck = raw

    # normalize keys (remove 'module.' prefix from distributed checkpoints)
    ck2 = {}
    for k, v in ck.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        ck2[nk] = v

    model_sd = model.state_dict()
    matched = {}
    skipped = []
    for k, v in ck2.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    if len(matched) == 0:
        # fallback to strict=False load to let PyTorch try its best
        try:
            model.load_state_dict(ck2, strict=False)
        except Exception:
            # final fallback: try original raw load and ignore errors
            try:
                model.load_state_dict(raw, strict=False)
            except Exception:
                pass
    else:
        # update model state dict with only matched params
        model_sd.update(matched)
        model.load_state_dict(model_sd)
    model.to(device).eval()

    img = Image.open(img_path).convert("RGB")
    t = torchvision.transforms.functional.to_tensor(img).to(device)
    with torch.no_grad():
        out = model([t])[0]

    boxes = out.get("boxes", torch.empty((0,4))).cpu().numpy().tolist()
    labels = out.get("labels", torch.empty((0,))).cpu().numpy().astype(int).tolist()
    scores = out.get("scores", torch.empty((0,))).cpu().numpy().tolist()

    preds = []
    for i,(b,l,s) in enumerate(zip(boxes, labels, scores), start=1):
        if s < conf: continue
        x1,y1,x2,y2 = [float(x) for x in b]
        w = x2-x1; h = y2-y1
        cname = None
        try:
            if class_names and 0 <= int(l) < len(class_names):
                cname = class_names[int(l)]
        except Exception:
            cname = None
        if not cname:
            cname = f"class_{int(l)}"
        preds.append({
            "id": i,
            "category_id": int(l),
            "category_name": cname,
            "bbox": [x1,y1,w,h],
            "score": float(s)
        })

    # draw image
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except:
        from PIL import ImageFont
        font = ImageFont.load_default()
    for p in preds:
        x,y,w,h = p["bbox"]
        draw.rectangle([x,y,x+w,y+h], outline="red", width=2)
        draw.text((x,y-12), f"{p['category_name']} {p['score']:.2f}", fill="white", font=font)

    Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
    draw_img.save(out_img_path)

    meta = {
        "input_path": str(img_path),
        "model_path": str(model_path),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "num_detections": len(preds),
        "predictions": preds
    }
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
