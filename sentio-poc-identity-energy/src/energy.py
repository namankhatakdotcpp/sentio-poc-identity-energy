from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_facemesh = None


def _get_facemesh():
    global _facemesh
    if _facemesh is None:
        _facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _facemesh


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def compute_brightness(face_crop_bgr: np.ndarray) -> float:
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    return _clip_score(float(np.mean(gray) / 2.55))


def _ear(landmarks, indices, width: int, height: int) -> float:
    p1, p2, p3, p4, p5, p6 = [
        np.array([landmarks[i].x * width, landmarks[i].y * height], dtype=np.float32)
        for i in indices
    ]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4) + 1e-8
    return float(vertical / horizontal)


def compute_eye_openness(face_crop_bgr: np.ndarray, fallback: float = 50.0) -> float:
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return fallback

    h, w = face_crop_bgr.shape[:2]
    if min(h, w) < 32:
        return fallback

    probe = face_crop_bgr
    max_side = max(h, w)
    if max_side > 256:
        scale = 256.0 / float(max_side)
        probe = cv2.resize(
            face_crop_bgr,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    try:
        mesh = _get_facemesh()
        face_rgb = cv2.cvtColor(probe, cv2.COLOR_BGR2RGB)
        results = mesh.process(face_rgb)
        if not results.multi_face_landmarks:
            return fallback

        ph, pw = probe.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = _ear(landmarks, LEFT_EYE, pw, ph)
        right_ear = _ear(landmarks, RIGHT_EYE, pw, ph)
        ear = (left_ear + right_ear) / 2.0

        score = (ear - 0.10) / (0.40 - 0.10) * 100.0
        return _clip_score(score)
    except Exception:
        return fallback


def _crop_roi(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 8:
        return None
    return roi


def compute_motion(
    prev_frame_bgr: Optional[np.ndarray],
    curr_frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    prev_gray: Optional[np.ndarray] = None,
    curr_gray: Optional[np.ndarray] = None,
) -> float:
    if prev_frame_bgr is None:
        return 0.0

    if prev_gray is None:
        prev_roi = _crop_roi(prev_frame_bgr, bbox)
        prev_gray_roi = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY) if prev_roi is not None else None
    else:
        prev_gray_roi = _crop_roi(prev_gray, bbox)

    if curr_gray is None:
        curr_roi = _crop_roi(curr_frame_bgr, bbox)
        curr_gray_roi = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY) if curr_roi is not None else None
    else:
        curr_gray_roi = _crop_roi(curr_gray, bbox)

    if prev_gray_roi is None or curr_gray_roi is None:
        return 0.0

    if prev_gray_roi.shape != curr_gray_roi.shape:
        curr_gray_roi = cv2.resize(curr_gray_roi, (prev_gray_roi.shape[1], prev_gray_roi.shape[0]))

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_roi,
        curr_gray_roi,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    score = float(np.mean(mag) * 20.0)
    return _clip_score(score)


def compute_energy_signals(
    face_crop_bgr: np.ndarray,
    prev_frame_bgr: Optional[np.ndarray],
    curr_frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    eye_fallback: float = 50.0,
    prev_gray: Optional[np.ndarray] = None,
    curr_gray: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    return {
        "brightness": compute_brightness(face_crop_bgr),
        "eye_openness": compute_eye_openness(face_crop_bgr, fallback=eye_fallback),
        "motion": compute_motion(
            prev_frame_bgr,
            curr_frame_bgr,
            bbox,
            prev_gray=prev_gray,
            curr_gray=curr_gray,
        ),
    }
