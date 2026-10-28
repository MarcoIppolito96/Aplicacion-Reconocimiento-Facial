"""
Pipeline facial compartido entre las notebooks (1-pre-processing / 2-train) y el backend.
Usar SIEMPRE estas funciones en los dos lados evita el clásico "anda en la notebook pero no en la app".

Convenciones:
- Imágenes de entrada en BGR uint8 (como devuelve cv2.imread).
- Detector: SCRFD (det_10g.onnx de InsightFace) -> cajas + 5 landmarks.
- Alineación: transformación de similitud a la plantilla ArcFace (norm_crop) -> TAMANIO_CARA x TAMANIO_CARA.
- Normalización para el modelo: BGR->RGB, (x - 127.5) / 127.5, HWC->CHW, dimensión de batch.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils import face_align


def cargar_detector(ruta: str | Path, tamanio_entrada: int = 640, umbral_score: float = 0.5) -> SCRFD:
    # Se carga una sola vez; ctx_id=-1 -> CPU
    detector = SCRFD(model_file=str(ruta))
    detector.prepare(ctx_id=-1, input_size=(tamanio_entrada, tamanio_entrada), det_thresh=umbral_score)
    return detector


def detectar(detector: SCRFD, imagen_bgr: np.ndarray, tamanio_min_cara: int = 20):
    """Devuelve lista de (caja[x1,y1,x2,y2] float, score, landmarks(5,2) en coords de la imagen)."""
    cajas, landmarks = detector.detect(imagen_bgr, max_num=0, metric="default")
    caras = []
    for i, c in enumerate(cajas):
        x1, y1, x2, y2, score = c
        # Descarta caras chicas: no tienen detalle suficiente para un buen embedding
        if min(x2 - x1, y2 - y1) < tamanio_min_cara:
            continue
        puntos = landmarks[i] if landmarks is not None else None
        caras.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(score), puntos))
    return caras


def alinear(imagen_bgr: np.ndarray, landmarks: np.ndarray, tamanio_cara: int = 112) -> np.ndarray:
    # Lleva ojos/nariz/boca a posiciones fijas (plantilla ArcFace) -> recorte alineado BGR
    return face_align.norm_crop(imagen_bgr, landmark=np.asarray(landmarks, dtype=np.float32), image_size=tamanio_cara)


def preprocesar(cara_bgr: np.ndarray) -> np.ndarray:
    """Recorte alineado BGR uint8 -> tensor float32 (1, 3, H, W) normalizado a [-1, 1]."""
    rgb = cv2.cvtColor(cara_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (rgb - 127.5) / 127.5
    return np.transpose(x, (2, 0, 1))[None, ...]


def normalizar_l2(vector: np.ndarray) -> np.ndarray:
    # Deja el vector con largo 1: así la similitud coseno es un simple producto punto
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norma = np.linalg.norm(vector)
    return vector / norma if norma > 0 else vector
