import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURACIÓN
# =========================

INPUT_IMG_DIR = "data/images"
INPUT_LABEL_DIR = "data/labels"

OUTPUT_IMG_DIR = "data/images_preprocessed"
OUTPUT_LABEL_DIR = "data/labels_preprocessed"

USE_BILATERAL = True   # recomendado solo si hay ruido de sensor
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

IMG_EXTENSIONS = [".jpg", ".png", ".jpeg"]


# =========================
# FUNCIONES DE PROCESADO
# =========================

def gray_world_white_balance(img):
    """
    White balance Gray-World.
    Justificado en literatura underwater 2022–2024.
    """
    img = img.astype(np.float32)
    mean_b, mean_g, mean_r = cv2.mean(img)[:3]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    img[:, :, 0] *= mean_gray / (mean_b + 1e-6)
    img[:, :, 1] *= mean_gray / (mean_g + 1e-6)
    img[:, :, 2] *= mean_gray / (mean_r + 1e-6)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def apply_clahe_luminance(img):
    """
    CLAHE SOLO en luminancia (LAB).
    Evita distorsionar color de pellets.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    img_out = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return img_out


def apply_bilateral(img):
    """
    Filtro bilateral suave.
    Mantiene bordes pequeños (pellets).
    NO recomendado para tiempo real.
    """
    return cv2.bilateralFilter(
        img,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE
    )


# =========================
# PIPELINE PRINCIPAL
# =========================

def preprocess_image(img):
    img = gray_world_white_balance(img)
    img = apply_clahe_luminance(img)

    if USE_BILATERAL:
        img = apply_bilateral(img)

    return img


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    ensure_dir(OUTPUT_IMG_DIR)
    ensure_dir(OUTPUT_LABEL_DIR)

    image_files = []
    for ext in IMG_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_IMG_DIR, f"*{ext}")))

    print(f"[INFO] Procesando {len(image_files)} imágenes")

    for img_path in tqdm(image_files):
        img_name = os.path.basename(img_path)
        name, _ = os.path.splitext(img_name)

        label_path = os.path.join(INPUT_LABEL_DIR, name + ".txt")

        # Leer imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] No se pudo leer {img_path}")
            continue

        # Preprocesar
        img_proc = preprocess_image(img)

        # Guardar imagen
        out_img_path = os.path.join(OUTPUT_IMG_DIR, img_name)
        cv2.imwrite(out_img_path, img_proc)

        # Copiar label (NO se modifica)
        if os.path.exists(label_path):
            out_label_path = os.path.join(OUTPUT_LABEL_DIR, name + ".txt")
            with open(label_path, "r") as f_in, open(out_label_path, "w") as f_out:
                f_out.write(f_in.read())
        else:
            print(f"[WARN] Label no encontrado para {img_name}")

    print("[INFO] Preprocesamiento completado")


if __name__ == "__main__":
    main()