import os
import json
from flask import Flask, render_template, request, send_file, redirect, url_for
from datetime import datetime
import csv
from pathlib import Path

from scripts.infer_yolo import infer_yolo
from scripts.infer_faster import infer_faster
from scripts.infer_ssd import infer_ssd
from scripts.infer_effdet import infer_effdet

app = Flask(__name__)
BASE_DIR = Path(__file__).parent.resolve()
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "static" / "uploads")
app.config["OUTPUT_FOLDER"] = str(BASE_DIR / "static" / "outputs")
app.config["MODEL_FOLDER"] = str(BASE_DIR / "models")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

# Map available models -> model filename under /models
MODELS = {
    "faster": {
        "aguacate": "faster_aguacate.pth",
        "mango": "faster_mango.pth"
    },
    "yolo": {
        "aguacate": "yolo_aguacate.pt",
        "mango": "yolo_mango.pt"
    },
    "ssd": {
        "aguacate": "ssd_aguacate.pth",
        "mango": "ssd_mango.pth"
    },
    "effdet": {
        "aguacate": "effdet_aguacate.pth",
        "mango": "effdet_mango.pth"
    }
}

# Human-readable display names for the model keys used by the app.
# Keep keys lowercase and values as the pretty names shown in the UI.
MODEL_DISPLAY = {
    "yolo": "YOLOv8",
    "effdet": "EfficientDet",
    "faster": "Faster R-CNN",
    "ssd": "MobileNetV3 SSD"
}

# Optional explicit class name mappings per model type and fruit.
# Each entry should be a list where index == class_id (0 is background).
# If a mapping is not provided for a model/fruit, a default of
# ["background", "<fruit>", "<fruit>_defecto"] is used.
MODEL_CLASS_NAMES = {
    # Provide standardized class name lists per model and fruit.
    # Index 0 == background, 1 == optimal, 2 == defect.
    "faster": {
        "aguacate": ["background", "aguacate_defecto", "aguacate_optimo"], # CORREGIDO
        "mango": ["background", "mango_defecto", "mango_optimo"]
    },
    "ssd": {
        "aguacate": ["background", "aguacate_defecto", "aguacate_optimo"], # CORREGIDO
        "mango": ["background", "mango_defecto", "mango_optimo"]
    },
    "effdet": {
        "aguacate": ["background", "aguacate_defecto", "aguacate_optimo"], # CORREGIDO
        "mango": ["background", "mango_defecto", "mango_optimo"]
    },
    "yolo": {
        "aguacate": ["background", "aguacate_optimo", "aguacate_defecto"],
        "mango": ["background", "mango_optimo", "mango_defecto"]
    }
}


@app.route("/")
def index():
    # pass a concrete list to the template and use lowercase fruit names
    return render_template(
        "index.html",
        models=list(MODELS.keys()),
        fruits=["aguacate", "mango"],
        model_display=MODEL_DISPLAY,
    )


@app.route("/predict", methods=["POST"])
def predict():
    model_type = request.form.get("model")
    fruit_type = request.form.get("fruit")
    file = request.files.get("image")

    # log incoming form values for debugging
    app.logger.debug("Predict request - model: %r, fruit: %r, file: %r", model_type, fruit_type, getattr(file, 'filename', None))

    if not file or file.filename == "":
        return redirect(url_for("index"))

    # save upload
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_name = f"{timestamp}_{file.filename}"
    upload_path = Path(app.config["UPLOAD_FOLDER"]) / safe_name
    file.save(str(upload_path))

    # pick model path
    model_filename = MODELS.get(model_type, {}).get(fruit_type)
    if model_filename is None:
        msg = f"Model not available for model={model_type!r} fruit={fruit_type!r}"
        app.logger.warning(msg)
        return msg, 400
    model_path = Path(app.config["MODEL_FOLDER"]) / model_filename
    if not model_path.exists():
        return f"Model file not found: {model_path}", 500

    # output paths
    out_img = Path(app.config["OUTPUT_FOLDER"]) / f"out_{safe_name}"
    out_json = out_img.with_suffix(".json")

    # run appropriate infer function
    try:
        # determine class name list: prefer explicit mapping, fallback to default
        default_names = ["background", fruit_type, f"{fruit_type}_defecto"]
        class_names = MODEL_CLASS_NAMES.get(model_type, {}).get(fruit_type, default_names)
        if model_type == "yolo":
            metadata = infer_yolo(str(model_path), str(upload_path), str(out_img), str(out_json))
        elif model_type == "faster":
            metadata = infer_faster(str(model_path), str(upload_path), str(out_img), str(out_json), class_names=class_names)
        elif model_type == "ssd":
            metadata = infer_ssd(str(model_path), str(upload_path), str(out_img), str(out_json), class_names=class_names)
        elif model_type == "effdet":
            # Use a lower default confidence for EfficientDet because some exported
            # checkpoints produce lower raw scores; keep as parameter for later UI exposure.
            metadata = infer_effdet(str(model_path), str(upload_path), str(out_img), str(out_json), conf=0.03, class_names=class_names)
        else:
            return "Unknown model", 400
    except Exception as e:
        return f"Inference failed: {e}", 500

    # serve web-friendly relative paths for templates (use forward slashes)
    # templates will call url_for('static', filename=...) so provide paths
    original_rel = f"uploads/{safe_name}"
    result_rel = f"outputs/{out_img.name}"
    json_rel = f"outputs/{out_json.name}"

    # model display name for the result page
    model_display_name = MODEL_DISPLAY.get(model_type, model_type.capitalize())

    return render_template("result.html",
                           original=original_rel,
                           result=result_rel,
                           metadata=metadata,
                           json_path=json_rel,
                           model_display=model_display_name)


@app.route("/download_csv")
def download_csv():
    # Expect a relative file name under the outputs folder, e.g. "outputs/xxx.json" or just the filename
    file_rel = request.args.get("file")
    if not file_rel:
        return "Missing file param", 400
    # normalize to basename to avoid directory traversal
    json_name = Path(file_rel).name
    json_path = Path(app.config["OUTPUT_FOLDER"]) / json_name
    if not json_path.exists():
        return "JSON not found", 404

    csv_path = json_path.with_suffix(".csv")
    with open(json_path, "r", encoding="utf8") as f:
        data = json.load(f)

    with open(csv_path, "w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "category", "score", "x", "y", "w", "h"])
        for p in data.get("predictions", []):
            writer.writerow([
                p.get("id"), p.get("category_name"), p.get("score"),
                p.get("bbox")[0], p.get("bbox")[1], p.get("bbox")[2], p.get("bbox")[3]
            ])
    return send_file(str(csv_path), as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
