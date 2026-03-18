from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np

from embedding_model import get_embedding_safe


ImageDB = Dict[str, List[np.ndarray]]


def _is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clean_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return stem.replace("_", " ").strip()


def load_known_faces(folder_path: str) -> ImageDB:
    """
    Load one-or-more reference embeddings per identity from a folder using DeepFace.
    Edge cases:
    - DeepFace fails: image is skipped with warning
    - Multiple faces: largest face is used with warning
    """
    known_faces: ImageDB = {}

    if not os.path.isdir(folder_path):
        warnings.warn(f"Known faces directory not found: {folder_path}")
        return known_faces

    files = sorted([f for f in os.listdir(folder_path) if _is_image_file(f)])
    if not files:
        warnings.warn(f"No image files found in known faces directory: {folder_path}")
        return known_faces

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        person_name = _clean_name(filename)

        try:
            image_bgr = cv2.imread(file_path)
            if image_bgr is None:
                warnings.warn(f"Failed to read image: {filename}")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            result = get_embedding_safe(image_rgb)

            if not result:
                warnings.warn(f"No face detected in {filename}. Skipping this image.")
                continue

            if len(result) > 1:
                warnings.warn(
                    f"Multiple faces found in {filename}. Using first detected face for {person_name}."
                )

            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            if person_name not in known_faces:
                known_faces[person_name] = []
            known_faces[person_name].append(embedding)

        except Exception as e:
            warnings.warn(f"Failed to process {filename}: {str(e)}")
            continue

    return known_faces


def flatten_known_faces(known_faces: ImageDB) -> Tuple[np.ndarray, List[str]]:
    """
    Convert dict[name -> [encodings...]] into aligned arrays for vectorized similarity.
    Returns:
        enc_matrix: shape (N, 128)
        labels: length N
    """
    encodings: List[np.ndarray] = []
    labels: List[str] = []

    for name, vectors in known_faces.items():
        for vec in vectors:
            encodings.append(vec.astype(np.float32))
            labels.append(name)

    if not encodings:
        return np.empty((0, 128), dtype=np.float32), []

    return np.vstack(encodings).astype(np.float32), labels
