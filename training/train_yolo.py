from ultralytics import YOLO
import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Ruta al data.yaml del dataset")
    parser.add_argument("--name", type=str, required=True,
                        help="Nombre del experimento (ej: aguacate_yolov8m)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: 'auto' (default), 'cpu', or CUDA id(s) like '0' or '0,1')")
    parser.add_argument("--project", type=str, default="runs_yolo",
                        help="Carpeta donde guardar los runs")
    args = parser.parse_args()

    # ------------------------------
    # CONFIGURACIÓN FIJA DEL MODELO
    # ------------------------------
    MODEL_NAME = "yolov8m.pt"     # Modelo que usarás SIEMPRE en este script
    EPOCHS = 100                  # Número de epochs
    BATCH = 8                     # Batch size ideal para YOLOv8m con GPU 8GB+
    IMG_SIZE = 640                # Tamaño de entrada
    PATIENCE = 8                  # Early stop
    # DEVICE selection: will be resolved from CLI arg 'args.device' below
    AUGMENT = True                # Activar data augmentation automático
    # ------------------------------

    print("\n========== CONFIGURACIÓN ==========")
    print(f"Modelo: {MODEL_NAME}")
    print(f"Dataset YAML: {args.data}")
    print(f"Experimento: {args.name}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH}")
    # Resolve device selection (auto -> GPU if available, else CPU)
    if getattr(args, 'device', 'auto') == "auto":
        try:
            args.device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    print(f"Patience: {PATIENCE}")
    print(f"Device: {args.device}")
    print(f"Output: {args.project}/{args.name}")
    print("===================================\n")

    # Cargar modelo
    model = YOLO(MODEL_NAME)

    # Entrenar
    model.train(
        data=args.data,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        project=args.project,
        name=args.name,
        device=args.device,
        augment=AUGMENT
    )

    print("\nEntrenamiento finalizado.")
    print(f"Revisa: {args.project}/{args.name}/weights/best.pt")

if __name__ == "__main__":
    main()
