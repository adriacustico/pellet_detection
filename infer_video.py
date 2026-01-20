import cv2
import argparse
from ultralytics import YOLO
from sort_tracker import Sort

# =========================
# CONFIGURACIÓN
# =========================

CONF_THRES = 0.05        # bajo para alto recall
IOU_THRES = 0.5
IMGSZ = 960

PELLET_CLASS_ID = 1     # AJUSTA si cambia dataset.yaml

# =========================
# INFERENCIA + TRACKING
# =========================

def run_video_inference(video_path, model_path, output_path=None):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # -------------------------
    # Inicializar SORT
    # -------------------------
    tracker = Sort(
        max_age=5,        # tolera desapariciones cortas
        iou_threshold=0.3
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------
        # YOLO detección
        # -------------------------
        results = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls == PELLET_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2, y2])

        # -------------------------
        # SORT tracking
        # -------------------------
        tracks = tracker.update(detections)

        # -------------------------
        # Visualización
        # -------------------------
        for trk in tracks:
            x1, y1, x2, y2 = map(int, trk.bbox)
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"ID {trk.id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        cv2.putText(
            frame,
            f"Pellets (tracks): {len(tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        if writer:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    print(f"[INFO] Video procesado: {frame_idx} frames")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferencia YOLOv8 + SORT para detección de pellets"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Ruta al video de entrada"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al modelo YOLO entrenado (.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta para guardar el video anotado"
    )

    args = parser.parse_args()

    run_video_inference(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output
    )
