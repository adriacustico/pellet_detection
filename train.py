from ultralytics import YOLO
import argparse
import yaml
import os


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    # -----------------------------
    # Cargar configuraciones
    # -----------------------------
    train_cfg = load_yaml(args.train_config)

    model_path = train_cfg.get("model", "models/yolov8n.pt")
    data_yaml = train_cfg.get("data", "data/dataset.yaml")

    # -----------------------------
    # Crear modelo
    # -----------------------------
    model = YOLO(model_path)

    # -----------------------------
    # Entrenamiento
    # -----------------------------
    model.train(
        data=data_yaml,
        epochs=train_cfg.get("epochs", 200),
        imgsz=train_cfg.get("imgsz", 960),
        batch=train_cfg.get("batch", 8),
        optimizer=train_cfg.get("optimizer", "AdamW"),
        lr0=train_cfg.get("lr0", 0.001),
        lrf=train_cfg.get("lrf", 0.1),
        weight_decay=train_cfg.get("weight_decay", 5e-4),
        patience=train_cfg.get("patience", 40),
        freeze=train_cfg.get("freeze", 10),
        workers=train_cfg.get("workers", 4),
        device=train_cfg.get("device", 0),
        project=train_cfg.get("project", "runs/train"),
        name=train_cfg.get("name", "pellet_yolov8"),
        cfg="configs/yolov8_custom.yaml",
        exist_ok=True,
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv8 - Detección de pellets")
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train.yaml",
        help="Ruta al archivo de configuración de entrenamiento"
    )
    args = parser.parse_args()
    main(args)
