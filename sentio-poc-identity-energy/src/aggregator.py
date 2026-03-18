from __future__ import annotations

import base64
import os
from typing import Dict, Iterable, List

import cv2
import numpy as np


def _energy_score(brightness: float, eye_openness: float, motion: float) -> float:
    return float((brightness * 0.35) + (eye_openness * 0.30) + (motion * 0.35))


def _laplacian_variance(image_bgr: np.ndarray) -> float:
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _encode_b64(image_bgr: np.ndarray, size=(240, 240)) -> str:
    if image_bgr is None or image_bgr.size == 0:
        image_bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def verdict_from_score(score: float) -> str:
    if score >= 70.0:
        return "HIGH"
    if score >= 40.0:
        return "MEDIUM"
    return "LOW"


def aggregate_people(
    frame_records: Iterable[Dict],
    profile_dir: str,
    id_prefix: str = "SM_P",
) -> List[Dict]:
    """
    One-pass aggregation to avoid storing large per-person frame lists in memory.
    Input record keys:
      name, crop, frame_idx, brightness, eye_openness, motion,
      match_confidence, track_confidence
    """
    os.makedirs(profile_dir, exist_ok=True)

    stats: Dict[str, Dict] = {}
    for rec in frame_records:
        name = rec["name"]
        frame_idx = int(rec["frame_idx"])
        crop = rec["crop"]

        if name not in stats:
            stats[name] = {
                "count": 0,
                "brightness_sum": 0.0,
                "eye_sum": 0.0,
                "motion_sum": 0.0,
                "match_conf_sum": 0.0,
                "track_conf_sum": 0.0,
                "first_seen": frame_idx,
                "last_seen": frame_idx,
                "best_crop": crop,
                "best_sharpness": _laplacian_variance(crop),
                "matched_any": bool(rec.get("matched", False)),
            }

        row = stats[name]
        row["count"] += 1
        row["brightness_sum"] += float(rec["brightness"])
        row["eye_sum"] += float(rec["eye_openness"])
        row["motion_sum"] += float(rec["motion"])
        row["match_conf_sum"] += float(rec.get("match_confidence", 0.0))
        row["track_conf_sum"] += float(rec.get("track_confidence", 0.0))
        row["first_seen"] = min(row["first_seen"], frame_idx)
        row["last_seen"] = max(row["last_seen"], frame_idx)
        row["matched_any"] = row["matched_any"] or bool(rec.get("matched", False))

        sharpness = _laplacian_variance(crop)
        if sharpness > row["best_sharpness"]:
            row["best_sharpness"] = sharpness
            row["best_crop"] = crop

    persons: List[Dict] = []
    for idx, (name, row) in enumerate(stats.items(), start=1):
        count = max(1, row["count"])
        avg_brightness = row["brightness_sum"] / count
        avg_eye = row["eye_sum"] / count
        avg_motion = row["motion_sum"] / count
        avg_match_conf = row["match_conf_sum"] / count
        avg_track_conf = row["track_conf_sum"] / count

        energy = float(np.clip(_energy_score(avg_brightness, avg_eye, avg_motion), 0.0, 100.0))
        best_crop = row["best_crop"]

        safe_name = name.replace(" ", "_")
        profile_path = os.path.join(profile_dir, f"{safe_name}.jpg")
        cv2.imwrite(profile_path, cv2.resize(best_crop, (240, 240), interpolation=cv2.INTER_CUBIC))

        persons.append(
            {
                "id": f"{id_prefix}{idx:04d}",
                "name": name,
                "energy_score": round(energy, 2),
                "brightness": round(avg_brightness, 2),
                "eye_openness": round(avg_eye, 2),
                "motion": round(avg_motion, 2),
                "frames": int(row["count"]),
                "time_range": [int(row["first_seen"]), int(row["last_seen"])],
                "profile_image_base64": _encode_b64(best_crop),
                "matched": bool(row["matched_any"] and not name.startswith("UNKNOWN_")),
                "verdict": verdict_from_score(energy),
                "avg_match_confidence": round(avg_match_conf, 3),
                "avg_track_confidence": round(avg_track_conf, 3),
            }
        )

    persons.sort(key=lambda p: (not p["matched"], -p["energy_score"]))
    for idx, person in enumerate(persons, start=1):
        person["id"] = f"{id_prefix}{idx:04d}"

    return persons
