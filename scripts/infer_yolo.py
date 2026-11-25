# scripts/infer_yolo.py
from ultralytics import YOLO
import json
from datetime import datetime
from pathlib import Path

def infer_yolo(model_path, img_path, out_img_path, out_json_path, conf=0.25):
    model = YOLO(str(model_path))
    results = model(str(img_path), conf=conf)
    res = results[0]

    # parse outputs
    boxes = []
    preds = []
    names = res.names if hasattr(res, "names") else {}
    # res.boxes: iterate robustly
    try:
        xyxy_list = res.boxes.xyxy.cpu().numpy().tolist()
        confs = res.boxes.conf.cpu().numpy().tolist()
        classes = res.boxes.cls.cpu().numpy().astype(int).tolist()
    except Exception:
        # fallback via res.boxes
        xyxy_list, confs, classes = [], [], []
        for b in res.boxes:
            xyxy_list.append([float(x) for x in b.xyxy[0].tolist()])
            confs.append(float(b.conf[0].tolist()[0]))
            classes.append(int(b.cls[0].tolist()[0]))

    for i, (b, c, s) in enumerate(zip(xyxy_list, classes, confs), start=1):
        x1,y1,x2,y2 = b
        w = x2 - x1
        h = y2 - y1
        name = names.get(int(c), str(c))
        preds.append({
            "id": i,
            "category_id": int(c),
            "category_name": name,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(s)
        })

    # save annotated image
    annotated = res.plot()
    Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_img_path) if hasattr(annotated, "save") else annotated # ultralytics returns numpy sometimes
    # ultralytics result.plot() returns numpy array in some versions
    try:
        from PIL import Image
        if isinstance(annotated, (list, tuple)):
            annotated = annotated[0]
        Image.fromarray(annotated).save(out_img_path)
    except Exception:
        pass

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
