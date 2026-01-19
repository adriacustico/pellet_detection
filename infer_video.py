import cv2
from ultralytics import YOLO
import argparse
import os
from collections import deque

# =========================
# CONFIGURACIÓN
# =========================

CONF_THRES = 0.10          # umbral base de confianza
IOU_THRES = 0.5
MIN_PERSISTENCE = 2        # frames consecutivos mínimos
BUFFER_SIZE = 5            # buffer temporal

PELLET_CLASS_ID = 0        # según dataset.yaml


# =========================
# FILTRO TEMPORAL
# =========================

class TemporalFilter:
    """
    Filtro temporal simple:
    una detección debe persistir varios frames
    para ser considerada válida.
    """

    def __init__(self, min_persistence=2, buffer_size=5):
        self.min_persistence = min_persistence
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections):
        """
        detections: lista de bounding boxes actuales
        """
        self.buffer.append(detections)

        # Contar apariciones
        flat = [tuple(map(int, det)) for frame in self.buffer for det in frame]
        counts = {}

        for det in flat:
            counts[det] = counts.get(det, 0) + 1

        # Mantener solo las persistentes
        valid = [
            list(det) for det, c in counts.items()
            if c >= self.min_persistence
        ]

        return valid


# =========================
# INFERENCIA EN VIDEO
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

    temporal_filter = TemporalFilter(
        min_persistence=MIN_PERSISTENCE,
        buffer_size=BUFFER_SIZE
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )[0]

        pellets = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls == PELLET_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    pellets.append([x1, y1, x2, y2])

        # Filtro temporal
        pellets_filtered = temporal_filter.update(pellets)

        # Dibujar resultados
        for (x1, y1, x2, y2) in pellets_filtered:
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Pellets: {len(pellets_filtered)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        if writer:
            writer.write(frame)

        cv2.imshow("Pellet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia YOLOv8 en video - Pellets")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Ruta al video de entrada"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Ruta al modelo entrenado"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta para guardar el video de salida (opcional)"
    )

    args = parser.parse_args()

    run_video_inference(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output
    )
