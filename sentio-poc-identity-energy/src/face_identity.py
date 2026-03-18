from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np


_KNOWN_ENCODINGS: List[np.ndarray] = []
_KNOWN_NAMES: List[str] = []
_CACHE_FOLDER: str | None = None
_CACHE_READY = False

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DEFAULT_THRESHOLD = 0.45


def _valid_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


def _person_name_from_file(filename: str) -> str:
    return Path(filename).stem.strip()


def _pick_best_encoding(image: np.ndarray) -> np.ndarray | None:
    try:
        locations = face_recognition.face_locations(image, model="hog")
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)
        if not encodings:
            return None

        if len(encodings) == 1 or len(locations) != len(encodings):
            return encodings[0]

        areas = []
        for top, right, bottom, left in locations:
            areas.append(max(0, bottom - top) * max(0, right - left))
        best_idx = int(np.argmax(areas))
        return encodings[best_idx]
    except Exception:
        return None


def load_known_faces(known_faces_dir: str = "known_faces"):
    known_encodings = []
    known_names = []

    for file in os.listdir(known_faces_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(known_faces_dir, file)
            name = os.path.splitext(file)[0]

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            print(f"[DEBUG] {name} encodings: {len(encodings)}")

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
            else:
                print(f"[WARNING] No face found in {file}")

    print(f"[DEBUG] FINAL loaded: {known_names}")

    global _KNOWN_ENCODINGS, _KNOWN_NAMES, _CACHE_READY
    _KNOWN_ENCODINGS = known_encodings
    _KNOWN_NAMES = known_names
    _CACHE_READY = True

    return known_encodings, known_names


def recognize_face(
    face_image: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
    return_confidence: bool = False,
):
    """
    Recognize a cropped face against cached known identities.

    Returns:
      - name (str) when return_confidence=False
      - (name, confidence) when return_confidence=True
    """
    def _unknown():
        return ("UNKNOWN", 0.0) if return_confidence else "UNKNOWN"

    try:
        if face_image is None or not isinstance(face_image, np.ndarray) or face_image.size == 0:
            return _unknown()

        if not _CACHE_READY:
            load_known_faces()

        if not _KNOWN_ENCODINGS:
            return _unknown()

        print(f"[DEBUG] encodings count: {len(_KNOWN_ENCODINGS)}")

        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            return _unknown()

        query = encodings[0]
        distances = face_recognition.face_distance(_KNOWN_ENCODINGS, query)
        if distances is None or len(distances) == 0:
            return _unknown()

        print(f"[DEBUG] comparing with {len(_KNOWN_ENCODINGS)} known faces")

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        matches = face_recognition.compare_faces(
            _KNOWN_ENCODINGS,
            query,
            tolerance=threshold,
        )

        if best_idx < len(matches) and matches[best_idx] and best_distance <= threshold:
            name = _KNOWN_NAMES[best_idx]
            similarity = float(max(0.0, min(1.0, 1.0 - (best_distance / max(threshold, 1e-6)))))
            print(f"[DEBUG] similarity with {name}: {similarity:.3f}")
            print(f"[MATCH] {name} ({similarity:.2f})")
            return (name, similarity) if return_confidence else name

        return _unknown()
    except Exception:
        return _unknown()
