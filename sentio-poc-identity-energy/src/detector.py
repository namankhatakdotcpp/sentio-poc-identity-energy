from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN


_detector_instance: Optional[MTCNN] = None


def get_detector() -> MTCNN:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MTCNN()
    return _detector_instance


def extract_frames(
    video_path: str,
    skip_n: int = 5,
    max_frames: Optional[int] = None,
) -> Tuple[List[Tuple[int, np.ndarray]], float]:
    """
    Read video and keep every (skip_n + 1)-th frame.
    Uses grab/retrieve so skipped frames avoid full decode.
    Returns:
        frames: [(frame_idx, frame_bgr), ...]
        fps: source video fps (fallback 25.0)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 25.0

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 25.0

    frames: List[Tuple[int, np.ndarray]] = []
    frame_idx = 0
    stride = max(1, skip_n + 1)

    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            frame_idx += 1
            continue

        frames.append((frame_idx, frame))
        if max_frames is not None and len(frames) >= max_frames:
            break

        frame_idx += 1

    cap.release()
    return frames, fps


def _clip_bbox(x: int, y: int, w: int, h: int, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + max(0, w))
    y2 = min(frame_h, y + max(0, h))
    return x1, y1, x2, y2


def detect_faces(
    frame_bgr: np.ndarray,
    min_upscale_size: int = 112,
    min_confidence: float = 0.80,
) -> List[Dict]:
    """
    Detect faces using face_recognition (CNN-based)
    Returns list of dicts:
    - bbox: (x1, y1, x2, y2)
    - crop: face crop in BGR (upscaled if min dimension < min_upscale_size)
    - confidence: detector confidence (set to 1.0 for CNN)
    """
    import face_recognition

    # Ensure RGB
    if frame_bgr.shape[2] == 3:
        rgb = frame_bgr[:, :, ::-1]
    else:
        rgb = frame_bgr

    boxes = face_recognition.face_locations(rgb, model="cnn")

    parsed: List[Dict] = []
    for (top, right, bottom, left) in boxes:
        x1, y1, x2, y2 = left, top, right, bottom

        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        ch, cw = crop.shape[:2]
        if min(ch, cw) < min_upscale_size:
            scale = min_upscale_size / float(min(ch, cw))
            nw = max(min_upscale_size, int(round(cw * scale)))
            nh = max(min_upscale_size, int(round(ch * scale)))
            crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)

        parsed.append(
            {
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "confidence": 1.0,  # HOG doesn't provide confidence, set to 1.0
            }
        )

    return parsed
            
        

    return parsed
