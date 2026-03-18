from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

from embedding_model import get_embedding_safe


BBox = Tuple[int, int, int, int]


def create_matcher_state(
    window_size: int = 5,
    max_missed: int = 3,
    decay: float = 0.88,
    boost: float = 0.12,
    min_track_confidence: float = 0.15,
) -> Dict:
    return {
        "next_track_id": 1,
        "unknown_counter": 1,
        "window_size": window_size,
        "max_missed": max_missed,
        "decay": decay,
        "boost": boost,
        "min_track_confidence": min_track_confidence,
        "tracks": {},
        "identity_history": defaultdict(lambda: deque(maxlen=window_size)),
        "similarity_history": defaultdict(lambda: deque(maxlen=window_size)),
        "unknown_name_by_track": {},
    }


def _cosine_similarity_vectorized(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return np.dot(matrix_norm, query_norm).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _clip_bbox(bbox: BBox, frame_shape: Tuple[int, int, int]) -> BBox:
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / max(union, 1))


def _next_unknown_name(track_id: int, state: Dict) -> str:
    if track_id not in state["unknown_name_by_track"]:
        name = f"UNKNOWN_{state['unknown_counter']:03d}"
        state["unknown_name_by_track"][track_id] = name
        state["unknown_counter"] += 1
    return state["unknown_name_by_track"][track_id]


def _get_embedding_from_crop(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Extract embedding from face crop with robust edge case handling."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    
    h, w = crop_bgr.shape[:2]
    if h < 40 or w < 40:
        return None

    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = get_embedding_safe(crop_rgb)
        
        if result:
            return np.array(result[0]["embedding"], dtype=np.float32)
    except Exception:
        return None
    
    return None


def batch_encode_detections(
    frame_bgr: np.ndarray,
    detections: List[Dict],
    embedding_min_size: int,
    state: Dict,
) -> None:
    """Compute embeddings for detections, reusing track embeddings when available."""
    for det in detections:
        det["embedding"] = None

    if not detections:
        return

    # Try to match detections to existing tracks and reuse embeddings
    tracks = state.get("tracks", {})
    for det in detections:
        # Check if this detection might belong to an existing track
        for track_id, track in tracks.items():
            if track.get("embedding") is not None and track.get("bbox") is not None:
                iou = _iou(det["bbox"], track["bbox"])
                # If IOU is high, likely same person - reuse track embedding
                if iou > 0.3:
                    det["embedding"] = track["embedding"]
                    break

    # Compute embeddings for detections that don't have cached ones
    for det in detections:
        if det["embedding"] is None:
            det["embedding"] = _get_embedding_from_crop(det["crop"])


def _score_association(
    det: Dict,
    track: Dict,
    frame_diag: float,
    iou_weight: float,
    center_weight: float,
    embedding_weight: float,
) -> float:
    iou = _iou(det["bbox"], track["bbox"])

    dcx, dcy = _bbox_center(det["bbox"])
    tcx, tcy = track["center"]
    center_dist = float(np.hypot(dcx - tcx, dcy - tcy))
    center_score = 1.0 - min(1.0, center_dist / max(frame_diag, 1.0))

    emb_score = 0.5
    if det.get("embedding") is not None and track.get("embedding") is not None:
        emb_score = max(0.0, _cosine_similarity(det["embedding"], track["embedding"]))

    return (iou_weight * iou) + (center_weight * center_score) + (embedding_weight * emb_score)


def _associate_tracks(
    detections: List[Dict],
    state: Dict,
    frame_shape: Tuple[int, int, int],
    min_score: float,
    iou_weight: float,
    center_weight: float,
    embedding_weight: float,
) -> None:
    tracks = state["tracks"]

    for trk in tracks.values():
        trk["missed"] += 1
        trk["confidence"] *= state["decay"]

    h, w = frame_shape[:2]
    frame_diag = float(np.hypot(w, h))

    candidates = []
    for det_idx, det in enumerate(detections):
        for track_id, trk in tracks.items():
            if trk["missed"] > state["max_missed"]:
                continue
            score = _score_association(
                det,
                trk,
                frame_diag,
                iou_weight=iou_weight,
                center_weight=center_weight,
                embedding_weight=embedding_weight,
            )
            candidates.append((score, det_idx, track_id))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_dets = set()
    used_tracks = set()

    for score, det_idx, track_id in candidates:
        if score < min_score:
            break
        if det_idx in used_dets or track_id in used_tracks:
            continue

        det = detections[det_idx]
        trk = tracks[track_id]

        det["track_id"] = track_id
        trk["bbox"] = det["bbox"]
        trk["center"] = _bbox_center(det["bbox"])
        trk["missed"] = 0
        trk["confidence"] = min(1.0, trk["confidence"] + state["boost"])

        if det.get("embedding") is not None:
            if trk.get("embedding") is None:
                trk["embedding"] = det["embedding"]
            else:
                trk["embedding"] = (0.8 * trk["embedding"]) + (0.2 * det["embedding"])

        used_dets.add(det_idx)
        used_tracks.add(track_id)

    for idx, det in enumerate(detections):
        if idx in used_dets:
            continue

        track_id = state["next_track_id"]
        state["next_track_id"] += 1

        det["track_id"] = track_id
        tracks[track_id] = {
            "bbox": det["bbox"],
            "center": _bbox_center(det["bbox"]),
            "embedding": det.get("embedding"),
            "confidence": 0.55,
            "missed": 0,
        }

    stale_ids = [
        track_id
        for track_id, trk in tracks.items()
        if trk["missed"] > state["max_missed"] or trk["confidence"] < state["min_track_confidence"]
    ]
    for track_id in stale_ids:
        tracks.pop(track_id, None)
        state["identity_history"].pop(track_id, None)
        state["similarity_history"].pop(track_id, None)


def _match_identity(
    embedding: np.ndarray | None,
    known_embeddings: np.ndarray,
    known_labels: List[str],
    threshold: float,
) -> Tuple[str | None, float]:
    if embedding is None or known_embeddings.size == 0 or not known_labels:
        return None, 0.0

    sims = _cosine_similarity_vectorized(embedding, known_embeddings)
    if sims.size == 0:
        return None, 0.0

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim < threshold:
        return None, best_sim
    return known_labels[best_idx], best_sim


def match_and_smooth(
    frame_bgr: np.ndarray,
    detections: List[Dict],
    known_embeddings: np.ndarray,
    known_labels: List[str],
    state: Dict,
    threshold: float = 0.50,
    embedding_min_size: int = 112,
    association_min_score: float = 0.33,
    iou_weight: float = 0.45,
    center_weight: float = 0.25,
    embedding_weight: float = 0.30,
) -> List[Dict]:
    if not detections:
        for trk in state["tracks"].values():
            trk["missed"] += 1
            trk["confidence"] *= state["decay"]
        return []

    batch_encode_detections(frame_bgr, detections, embedding_min_size=embedding_min_size, state=state)

    _associate_tracks(
        detections,
        state,
        frame_shape=frame_bgr.shape,
        min_score=association_min_score,
        iou_weight=iou_weight,
        center_weight=center_weight,
        embedding_weight=embedding_weight,
    )

    for det in detections:
        track_id = det["track_id"]
        trk = state["tracks"].get(track_id)

        raw_label, raw_similarity = _match_identity(
            embedding=det.get("embedding"),
            known_embeddings=known_embeddings,
            known_labels=known_labels,
            threshold=threshold,
        )
        if raw_label is None:
            raw_label = _next_unknown_name(track_id, state)

        label_hist: Deque[str] = state["identity_history"][track_id]
        sim_hist: Deque[float] = state["similarity_history"][track_id]
        label_hist.append(raw_label)
        sim_hist.append(max(0.0, raw_similarity))

        stable_name = Counter(label_hist).most_common(1)[0][0]
        matched = not stable_name.startswith("UNKNOWN_")

        if matched:
            non_zero_sims = [x for x in sim_hist if x > 0.0]
            sim_score = float(np.mean(non_zero_sims)) if non_zero_sims else 0.0
            track_conf = trk["confidence"] if trk else 0.3
            match_conf = max(0.0, min(1.0, sim_score * track_conf))
        else:
            match_conf = 0.0

        det["name"] = stable_name
        det["matched"] = matched
        det["track_confidence"] = round(float(trk["confidence"] if trk else 0.0), 4)
        det["match_confidence"] = round(float(match_conf), 4)

    return detections
