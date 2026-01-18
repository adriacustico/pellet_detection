import albumentations as A
import cv2
import numpy as np


def get_augmentation_pipeline():
    """
    Pipeline de data augmentation físicamente coherente
    para detección de pellets en acuicultura.

    Basado en literatura 2022–2024:
    - NO flips
    - NO rotaciones arbitrarias
    - NO mosaic / mixup
    """

    return A.Compose(
        [
            # -----------------------------
            # Variaciones de iluminación
            # -----------------------------
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.4
            ),

            # -----------------------------
            # Motion blur direccional
            # Simula pellets en caída
            # -----------------------------
            A.MotionBlur(
                blur_limit=(3, 5),
                p=0.3
            ),

            # -----------------------------
            # Ruido suave (marine snow)
            # -----------------------------
            A.GaussNoise(
                var_limit=(5, 15),
                mean=0,
                p=0.2
            ),

            # -----------------------------
            # Turbidez simulada
            # Blur leve + pérdida de contraste
            # -----------------------------
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2
            ),

            # -----------------------------
            # Oclusión parcial (pellets tapados por peces)
            # -----------------------------
            A.CoarseDropout(
                max_holes=2,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=0.2
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3
        ),
    )


# -------------------------------------------------
# FUNCIÓN DE APLICACIÓN (para debug o integración)
# -------------------------------------------------

def apply_augmentation(image, bboxes, class_labels):
    """
    image: np.array (BGR)
    bboxes: lista de [x_center, y_center, w, h] (YOLO)
    class_labels: lista de clases
    """

    aug = get_augmentation_pipeline()
    augmented = aug(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )

    return augmented["image"], augmented["bboxes"], augmented["class_labels"]
