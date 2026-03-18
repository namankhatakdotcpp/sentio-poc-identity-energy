from __future__ import annotations

import threading

from deepface import DeepFace


MODEL = DeepFace.build_model("Facenet")
MODEL_LOCK = threading.Lock()


def get_embedding_safe(img_rgb: bytes | object) -> dict | None:
    """
    Thread-safe wrapper for DeepFace.represent().
    Prevents race conditions in async API mode.
    
    Args:
        img_rgb: RGB numpy array or image path
        
    Returns:
        DeepFace result dict or None on failure
    """
    try:
        with MODEL_LOCK:
            result = DeepFace.represent(
                img_path=img_rgb,
                model_name="Facenet",
                model=MODEL,
                enforce_detection=False,
            )
        return result
    except Exception:
        return None
