import cv2
from ultralytics import YOLO
import argparse
import numpy as np
from collections import deque

# =========================
# CONFIGURACIÓN
# =========================

CONF_THRES = 0.10
IOU_THRES = 0.5

PELLET_CLASS_ID = 0

TEMPORAL_WINDOW = 5       # frames
MIN_PERSISTENCE = 2       # detección válida si aparece >= N frames


# =========================
# FILTRO TEMPORAL
# =========================

class TemporalFilter:
    def __init__(self, window=5, min_persistence=2):
        self.window = window
        self.min_persistence = min_persistence
        self.buffer = deque(maxlen=window)

    def update(self, detections):
        self.buffer.append(detections)

        flat = [tuple(map(int, d)) for frame in self.buffer for d in frame]
        counts = {}

        for d in flat:
            counts[d] = counts.get(d, 0) + 1

        valid = [
            list(d) for d, c in counts.items()
            if c >= self.min_persistence
        ]

        return valid


# =========================
# MÉTRICAS DE VIDEO
# =========================

def compute_temporal_stability(counts):
    """
    Mide cuán estable es el conteo de pellets en el tiempo.
    CV = std / mean
    """
    counts = np.array(counts)
    if counts.mean() == 0:
        return 0.0
    return counts.std() / counts.mean()


def compute_false_positive_rate(counts, fps):
    """
    Falsos positivos aproximados:
    detecciones aisladas y no persistentes
    """
    isolated = sum(1 for c in counts if c == 1)
    duration_sec = len(counts) / fps
    return isolated / max(duration_sec, 1e-6)


# =========================
# EVALUACIÓN EN VIDEO
# =========================

def evaluate_video(video_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    temporal_filter = TemporalFilter(
        window=TEMPORAL_WINDOW,
        min_persistence=MIN_PERSISTENCE
    )

    raw_counts = []
    filtered_counts = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            imgsz=960,
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

        pellets_filtered = temporal_filter.update(pellets)

        raw_counts.append(len(pellets))
        filtered_counts.append(len(pellets_filtered))

        frame_idx += 1

    cap.release()

    # =========================
    # MÉTRICAS FINALES
    # =========================

    stability_raw = compute_temporal_stability(raw_counts)
    stability_filtered = compute_temporal_stability(filtered_counts)

    fp_rate_raw = compute_false_positive_rate(raw_counts, fps)
    fp_rate_filtered = compute_false_positive_rate(filtered_counts, fps)

    return {
        "frames": frame_idx,
        "fps": fps,
        "mean_raw_count": np.mean(raw_counts),
        "mean_filtered_count": np.mean(filtered_counts),
        "stability_raw": stability_raw,
        "stability_filtered": stability_filtered,
        "false_positives_raw_per_sec": fp_rate_raw,
        "false_positives_filtered_per_sec": fp_rate_filtered,
    }


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación temporal YOLOv8 - Pellets")
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

    args = parser.parse_args()

    metrics = evaluate_video(
        video_path=args.video,
        model_path=args.model
    )

    print("\n=== RESULTADOS DE EVALUACIÓN ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
