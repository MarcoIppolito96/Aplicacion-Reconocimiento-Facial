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
import onnxruntime as ort
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils import face_align

ort.set_default_logger_severity(3)  # oculta avisos internos de onnxruntime


def _proveedores(usar_gpu: bool) -> list[str]:
    # Con GPU intenta CUDA y, si no está disponible, sigue en CPU
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if usar_gpu else ["CPUExecutionProvider"]


def cargar_detector(ruta: str | Path, tamanio_entrada: int = 640, umbral_score: float = 0.5,
                    usar_gpu: bool = False) -> SCRFD:
    # Se carga una sola vez; ctx_id=-1 -> CPU, ctx_id=0 -> GPU
    sesion = ort.InferenceSession(str(ruta), providers=_proveedores(usar_gpu))
    detector = SCRFD(model_file=str(ruta), session=sesion)
    detector.prepare(ctx_id=0 if usar_gpu else -1, input_size=(tamanio_entrada, tamanio_entrada),
                     det_thresh=umbral_score)
    return detector


def cargar_reconocedor(ruta: str | Path, usar_gpu: bool = False) -> ort.InferenceSession:
    # Modelo de embeddings en formato ONNX (ArcFace o la CNN propia exportada)
    return ort.InferenceSession(str(ruta), providers=_proveedores(usar_gpu))


def detectar(detector: SCRFD, imagen_bgr: np.ndarray, tamanio_min_cara: int = 20,
             fraccion_borde: float = 0.0):
    """Devuelve lista de (caja[x1,y1,x2,y2] float, score, landmarks(5,2) en coords de la imagen)."""
    # Borde negro opcional: el detector falla si la cara ocupa toda la imagen (recortes ajustados)
    m = int(fraccion_borde * max(imagen_bgr.shape[:2]))
    if m > 0:
        imagen_bgr = cv2.copyMakeBorder(imagen_bgr, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    cajas, landmarks = detector.detect(imagen_bgr, max_num=0, metric="default")
    caras = []
    for i, c in enumerate(cajas):
        x1, y1, x2, y2, score = c
        # Descarta caras chicas: no tienen detalle suficiente para un buen embedding
        if min(x2 - x1, y2 - y1) < tamanio_min_cara:
            continue
        puntos = landmarks[i] - m if landmarks is not None else None  # vuelve a coords sin borde
        caja = np.array([x1, y1, x2, y2], dtype=np.float32) - m
        caras.append((caja, float(score), puntos))
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


def calcular_embeddings(sesion: ort.InferenceSession, caras_bgr: list[np.ndarray],
                        tamanio_lote: int = 64) -> np.ndarray:
    """Embeddings normalizados L2 de muchas caras alineadas, por lotes (más rápido que de a una)."""
    nombre_entrada = sesion.get_inputs()[0].name
    salida = []
    for i in range(0, len(caras_bgr), tamanio_lote):
        lote = np.concatenate([preprocesar(c) for c in caras_bgr[i:i + tamanio_lote]])
        emb = sesion.run(None, {nombre_entrada: lote})[0]
        salida.append(emb / np.linalg.norm(emb, axis=1, keepdims=True))
    return np.concatenate(salida).astype(np.float32)
