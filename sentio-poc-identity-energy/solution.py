from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aggregator import aggregate_people  # noqa: E402
from config import AppConfig, load_config  # noqa: E402
from detector import detect_faces, extract_frames  # noqa: E402
from encoder import flatten_known_faces, load_known_faces  # noqa: E402
from energy import compute_energy_signals  # noqa: E402
from face_identity import (  # noqa: E402
    load_known_faces as load_known_faces_identity,
    recognize_face,
)
from matcher import create_matcher_state, match_and_smooth  # noqa: E402
from reporter import write_csv, write_html_report, write_json  # noqa: E402


def _setup_logger(cfg: AppConfig) -> logging.Logger:
    logger = logging.getLogger("sentio.pipeline")
    if not logger.handlers:
        level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(cfg.logging.fmt))
        logger.addHandler(handler)
    return logger


def _log_event(logger: logging.Logger, event: str, **data) -> None:
    payload = {"event": event, **data}
    logger.info(json.dumps(payload, default=str))


def _energy_score(cfg: AppConfig, brightness: float, eye_openness: float, motion: float) -> float:
    return (
        brightness * cfg.energy.brightness_weight
        + eye_openness * cfg.energy.eye_weight
        + motion * cfg.energy.motion_weight
    )


def _draw_demo_frame(frame: np.ndarray, detections: List[Dict], fps: float = 0.0) -> np.ndarray:
    annotated = frame.copy()
    
    # Add FPS counter (top-left)
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Add system branding (bottom-left)
    cv2.putText(
        annotated,
        "SentioMind Face Recognition",
        (10, annotated.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 200, 0) if det.get("matched", False) else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        name = det.get("name", "UNKNOWN")
        label = f"{name} ({det.get('match_confidence', 0.0):.2f})"
        
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(18, y1 - 8)
        
        cv2.rectangle(
            annotated,
            (x1, text_y - text_h - 6),
            (x1 + text_w + 6, text_y + 4),
            color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _apply_identity_recognition_fallback(detections: List[Dict]) -> None:
    """
    Keep existing tracking/matching pipeline intact.
    Only relabel detections currently marked UNKNOWN_* when a known face match exists.
    """
    for det in detections:
        current_name = str(det.get("name", ""))
        if not current_name.startswith("UNKNOWN"):
            continue

        crop = det.get("crop")
        if crop is None:
            continue

        matched_name, identity_conf = recognize_face(
            crop,
            return_confidence=True,
        )
        if matched_name == "UNKNOWN":
            continue

        det["name"] = matched_name
        det["matched"] = True
        det["match_confidence"] = round(
            float(max(det.get("match_confidence", 0.0), identity_conf)),
            4,
        )


def _write_demo_video(
    annotated_frames: List[Tuple[int, np.ndarray]],
    output_path: str,
    fps: float,
) -> bool:
    if not annotated_frames:
        return False

    _, first = annotated_frames[0]
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 10.0,
        (w, h),
    )
    if not writer.isOpened():
        return False

    for _, frame in annotated_frames:
        writer.write(frame)
    writer.release()
    return True


def _process_frames(
    frames: List[Tuple[int, np.ndarray]],
    fps: float,
    known_embeddings: np.ndarray,
    known_labels: List[str],
    cfg: AppConfig,
    logger: logging.Logger,
    output_dir: str,
) -> Dict:
    profile_dir = os.path.join(output_dir, "profile_crops")
    os.makedirs(profile_dir, exist_ok=True)
    
    _log_event(logger, "processing_start", total_frames=len(frames), fps=fps)
    processing_start = time.time()

    matcher_state = create_matcher_state(
        window_size=cfg.matcher.window_size,
        max_missed=cfg.matcher.max_missed,
        decay=cfg.matcher.decay,
        boost=cfg.matcher.boost,
        min_track_confidence=cfg.matcher.min_track_confidence,
    )

    all_records: List[Dict] = []
    timeline_records: List[Dict] = []
    annotated_frames: List[Tuple[int, np.ndarray]] = []

    prev_frame = None
    prev_gray = None

    for idx, (frame_idx, frame) in enumerate(frames, start=1):
        frame_start = time.time()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detect_start = time.time()
        detections = detect_faces(
            frame,
            min_upscale_size=cfg.detector.min_upscale_size,
            min_confidence=cfg.detector.min_face_confidence,
        )
        print(f"[DEBUG] detections in frame: {len(detections)}")
        detect_time = time.time() - detect_start
        
        match_start = time.time()
        detections = match_and_smooth(
            frame_bgr=frame,
            detections=detections,
            known_embeddings=known_embeddings,
            known_labels=known_labels,
            state=matcher_state,
            threshold=cfg.matcher.similarity_threshold,
            embedding_min_size=cfg.matcher.embedding_min_size,
            association_min_score=cfg.matcher.association_min_score,
            iou_weight=cfg.matcher.iou_weight,
            center_weight=cfg.matcher.center_weight,
            embedding_weight=cfg.matcher.embedding_weight,
        )
        _apply_identity_recognition_fallback(detections)
        match_time = time.time() - match_start

        for det in detections:
            signals = compute_energy_signals(
                face_crop_bgr=det["crop"],
                prev_frame_bgr=prev_frame,
                curr_frame_bgr=frame,
                bbox=det["bbox"],
                eye_fallback=cfg.energy.eye_fallback,
                prev_gray=prev_gray,
                curr_gray=curr_gray,
            )
            det["energy_score"] = _energy_score(
                cfg,
                signals["brightness"],
                signals["eye_openness"],
                signals["motion"],
            )

            record = {
                "name": det["name"],
                "crop": det["crop"],
                "bbox": det["bbox"],
                "frame_idx": frame_idx,
                "brightness": signals["brightness"],
                "eye_openness": signals["eye_openness"],
                "motion": signals["motion"],
                "matched": det.get("matched", False),
                "match_confidence": det.get("match_confidence", 0.0),
                "track_confidence": det.get("track_confidence", 0.0),
            }
            all_records.append(record)
            timeline_records.append(
                {
                    "name": det["name"],
                    "frame_idx": frame_idx,
                    "match_confidence": det.get("match_confidence", 0.0),
                    "track_confidence": det.get("track_confidence", 0.0),
                }
            )
            
            # Memory management: prevent unbounded growth on long videos
            if len(all_records) > 5000:
                all_records.pop(0)
            if len(timeline_records) > 10000:
                timeline_records.pop(0)

        annotated_frames.append((frame_idx, _draw_demo_frame(frame, detections, fps=fps)))
        prev_frame = frame
        prev_gray = curr_gray
        
        frame_total_time = time.time() - frame_start

        if idx == 1 or idx % 10 == 0 or idx == len(frames):
            _log_event(
                logger,
                "frame_progress",
                processed=idx,
                total=len(frames),
                frame_idx=frame_idx,
                detections=len(detections),
                frame_time_ms=round(frame_total_time * 1000, 1),
                detect_time_ms=round(detect_time * 1000, 1),
                match_time_ms=round(match_time * 1000, 1),
            )

    processing_time = time.time() - processing_start
    _log_event(logger, "frame_processing_complete", time_sec=round(processing_time, 2))

    agg_start = time.time()
    persons = aggregate_people(all_records, profile_dir=profile_dir, id_prefix="SM_P")
    agg_time = time.time() - agg_start
    _log_event(logger, "aggregation_complete", persons_found=len(persons), time_sec=round(agg_time, 2))

    json_path = os.path.join(output_dir, "integration_output.json")
    csv_path = os.path.join(output_dir, "integration_output.csv")
    report_path = os.path.join(output_dir, "report.html")
    demo_path = os.path.join(output_dir, "demo.mp4")

    report_start = time.time()
    write_json(persons, json_path)
    write_csv(persons, csv_path)
    write_html_report(persons, report_path, timeline_records=timeline_records)
    report_time = time.time() - report_start
    _log_event(logger, "reports_written", time_sec=round(report_time, 2))

    demo_ok = False
    if cfg.runtime.write_demo_video:
        demo_start = time.time()
        demo_ok = _write_demo_video(annotated_frames, demo_path, fps=max(5.0, fps / 5.0))
        demo_time = time.time() - demo_start
        if demo_ok:
            _log_event(logger, "demo_video_written", time_sec=round(demo_time, 2))
        else:
            logger.warning("Failed to write demo video")

    return {
        "persons": persons,
        "json_path": json_path,
        "csv_path": csv_path,
        "report_path": report_path,
        "demo_path": demo_path,
        "demo_ok": demo_ok,
    }


def run_batch_pipeline(
    known_faces_dir: str,
    video_path: str,
    output_dir: str,
    cfg: AppConfig,
    logger: logging.Logger,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    _log_event(logger, "stage_start", stage="load_faces")
    known_faces = load_known_faces(known_faces_dir)
    known_embeddings, known_labels = flatten_known_faces(known_faces)
    identity_encodings, _ = load_known_faces_identity(known_faces_dir)
    _log_event(
        logger,
        "stage_done",
        stage="load_faces",
        identities=len(known_faces),
        embeddings=int(known_embeddings.shape[0]),
        identity_face_rec_refs=len(identity_encodings),
    )

    _log_event(logger, "stage_start", stage="extract_frames")
    frames, fps = extract_frames(
        video_path,
        skip_n=cfg.detector.skip_n,
        max_frames=cfg.detector.max_frames,
    )
    _log_event(
        logger,
        "stage_done",
        stage="extract_frames",
        extracted_frames=len(frames),
        source_fps=round(float(fps), 2),
    )

    _log_event(logger, "stage_start", stage="process_frames")
    result = _process_frames(
        frames=frames,
        fps=fps,
        known_embeddings=known_embeddings,
        known_labels=known_labels,
        cfg=cfg,
        logger=logger,
        output_dir=output_dir,
    )
    _log_event(logger, "stage_done", stage="process_frames", persons=len(result["persons"]))
    return result


def run_realtime_pipeline(
    known_faces_dir: str,
    video_source: str,
    output_dir: str,
    cfg: AppConfig,
    logger: logging.Logger,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    profile_dir = os.path.join(output_dir, "profile_crops")
    os.makedirs(profile_dir, exist_ok=True)

    _log_event(logger, "stage_start", stage="load_faces")
    known_faces = load_known_faces(known_faces_dir)
    known_embeddings, known_labels = flatten_known_faces(known_faces)
    identity_encodings, _ = load_known_faces_identity(known_faces_dir)
    _log_event(
        logger,
        "stage_done",
        stage="load_faces",
        identities=len(known_faces),
        embeddings=int(known_embeddings.shape[0]),
        identity_face_rec_refs=len(identity_encodings),
    )

    source = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open realtime source: {video_source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 25.0

    matcher_state = create_matcher_state(
        window_size=cfg.matcher.window_size,
        max_missed=cfg.matcher.max_missed,
        decay=cfg.matcher.decay,
        boost=cfg.matcher.boost,
        min_track_confidence=cfg.matcher.min_track_confidence,
    )

    all_records: List[Dict] = []
    timeline_records: List[Dict] = []
    annotated_frames: List[Tuple[int, np.ndarray]] = []

    prev_frame = None
    prev_gray = None
    frame_idx = 0
    processed = 0
    stride = max(1, cfg.detector.skip_n + 1)

    _log_event(logger, "stage_start", stage="realtime_loop", source=str(video_source))
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_faces(
            frame,
            min_upscale_size=cfg.detector.min_upscale_size,
            min_confidence=cfg.detector.min_face_confidence,
        )
        detections = match_and_smooth(
            frame_bgr=frame,
            detections=detections,
            known_embeddings=known_embeddings,
            known_labels=known_labels,
            state=matcher_state,
            threshold=cfg.matcher.similarity_threshold,
            embedding_min_size=cfg.matcher.embedding_min_size,
            association_min_score=cfg.matcher.association_min_score,
            iou_weight=cfg.matcher.iou_weight,
            center_weight=cfg.matcher.center_weight,
            embedding_weight=cfg.matcher.embedding_weight,
        )
        _apply_identity_recognition_fallback(detections)

        for det in detections:
            signals = compute_energy_signals(
                face_crop_bgr=det["crop"],
                prev_frame_bgr=prev_frame,
                curr_frame_bgr=frame,
                bbox=det["bbox"],
                eye_fallback=cfg.energy.eye_fallback,
                prev_gray=prev_gray,
                curr_gray=curr_gray,
            )
            all_records.append(
                {
                    "name": det["name"],
                    "crop": det["crop"],
                    "bbox": det["bbox"],
                    "frame_idx": frame_idx,
                    "brightness": signals["brightness"],
                    "eye_openness": signals["eye_openness"],
                    "motion": signals["motion"],
                    "matched": det.get("matched", False),
                    "match_confidence": det.get("match_confidence", 0.0),
                    "track_confidence": det.get("track_confidence", 0.0),
                }
            )
            timeline_records.append(
                {
                    "name": det["name"],
                    "frame_idx": frame_idx,
                    "match_confidence": det.get("match_confidence", 0.0),
                    "track_confidence": det.get("track_confidence", 0.0),
                }
            )

        annotated = _draw_demo_frame(frame, detections, fps=fps)
        annotated_frames.append((frame_idx, annotated))

        if not cfg.runtime.no_display:
            cv2.imshow("Sentio Realtime", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        prev_frame = frame
        prev_gray = curr_gray
        processed += 1
        frame_idx += 1

        if cfg.detector.max_frames is not None and processed >= cfg.detector.max_frames:
            break

    cap.release()
    if not cfg.runtime.no_display:
        cv2.destroyAllWindows()

    persons = aggregate_people(all_records, profile_dir=profile_dir, id_prefix="SM_P")

    json_path = os.path.join(output_dir, "integration_output.json")
    csv_path = os.path.join(output_dir, "integration_output.csv")
    report_path = os.path.join(output_dir, "report.html")
    demo_path = os.path.join(output_dir, "demo.mp4")

    write_json(persons, json_path)
    write_csv(persons, csv_path)
    write_html_report(persons, report_path, timeline_records=timeline_records)

    demo_ok = False
    if cfg.runtime.write_demo_video:
        demo_ok = _write_demo_video(annotated_frames, demo_path, fps=max(5.0, fps / 5.0))

    _log_event(logger, "stage_done", stage="realtime_loop", processed_frames=processed, persons=len(persons))

    return {
        "persons": persons,
        "json_path": json_path,
        "csv_path": csv_path,
        "report_path": report_path,
        "demo_path": demo_path,
        "demo_ok": demo_ok,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentio POC Identity + Energy pipeline")
    parser.add_argument("--known_faces", default=os.path.join(ROOT_DIR, "known_faces"))
    parser.add_argument("--video", default=os.path.join(ROOT_DIR, "data", "video_sample_1.mov"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT_DIR, "outputs"))
    parser.add_argument("--config", default=None, help="Optional JSON config override path.")
    parser.add_argument("--realtime", action="store_true", help="Enable realtime source mode (camera or live stream).")
    parser.add_argument("--no_display", action="store_true", help="Disable UI window in realtime mode.")
    parser.add_argument("--skip_n", type=int, default=None, help="Override config detector.skip_n")
    parser.add_argument("--max_frames", type=int, default=None, help="Override config detector.max_frames")
    parser.add_argument("--threshold", type=float, default=None, help="Override config matcher.similarity_threshold")
    return parser.parse_args()


def _apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    if args.skip_n is not None:
        cfg.detector.skip_n = args.skip_n
    if args.max_frames is not None:
        cfg.detector.max_frames = args.max_frames
    if args.threshold is not None:
        cfg.matcher.similarity_threshold = args.threshold
    if args.realtime:
        cfg.runtime.realtime = True
    if args.no_display:
        cfg.runtime.no_display = True


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    logger = _setup_logger(cfg)

    start = time.time()
    _log_event(logger, "pipeline_start", realtime=cfg.runtime.realtime)

    if cfg.runtime.realtime:
        result = run_realtime_pipeline(
            known_faces_dir=args.known_faces,
            video_source=args.video,
            output_dir=args.output_dir,
            cfg=cfg,
            logger=logger,
        )
    else:
        result = run_batch_pipeline(
            known_faces_dir=args.known_faces,
            video_path=args.video,
            output_dir=args.output_dir,
            cfg=cfg,
            logger=logger,
        )

    elapsed = time.time() - start
    _log_event(
        logger,
        "pipeline_done",
        runtime_sec=round(elapsed, 3),
        persons=len(result["persons"]),
        report=result["report_path"],
        json=result["json_path"],
        csv=result["csv_path"],
        demo=result["demo_path"],
        demo_ok=result["demo_ok"],
    )


if __name__ == "__main__":
    main()
