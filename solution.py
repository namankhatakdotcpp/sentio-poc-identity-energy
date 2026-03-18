import argparse
import base64
import json
import os
import time
import warnings
from collections import Counter, defaultdict, deque
from datetime import datetime

import cv2
import face_recognition
import mediapipe as mp
import numpy as np


LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
GRID_ROWS = 4
GRID_COLS = 4
SCHOOL_NAME = "Sentio Mind (IIT Mandi)"
SCHOOL_PREFIX_DEFAULT = "SM"

_track_history = defaultdict(lambda: deque(maxlen=5))
_cell_unknown_name = {}
_unknown_counter = 1
_facemesh_instance = None


def encode_b64(image_bgr, size=(240, 240)):
    if image_bgr is None or image_bgr.size == 0:
        image_bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def verdict(score):
    if score >= 70.0:
        return "high"
    if score >= 40.0:
        return "moderate"
    return "low"


def _clip_0_100(value):
    return float(np.clip(value, 0.0, 100.0))


def _cosine_distance(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - float(np.dot(a, b) / denom)


def _laplacian_variance(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _get_facemesh():
    global _facemesh_instance
    if _facemesh_instance is None:
        _facemesh_instance = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _facemesh_instance


def _ear(landmarks, idxs, width, height):
    p1, p2, p3, p4, p5, p6 = [
        np.array([landmarks[i].x * width, landmarks[i].y * height], dtype=np.float32)
        for i in idxs
    ]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = (2.0 * np.linalg.norm(p1 - p4)) + 1e-8
    return float(vertical / horizontal)


def _grid_cell(bbox, frame_shape):
    top, right, bottom, left = bbox
    h, w = frame_shape[:2]
    cx = max(0, min(w - 1, (left + right) // 2))
    cy = max(0, min(h - 1, (top + bottom) // 2))
    cell_w = max(1, w // GRID_COLS)
    cell_h = max(1, h // GRID_ROWS)
    col = min(GRID_COLS - 1, cx // cell_w)
    row = min(GRID_ROWS - 1, cy // cell_h)
    return int(row), int(col)


def _get_unknown_for_cell(cell):
    global _unknown_counter
    if cell not in _cell_unknown_name:
        _cell_unknown_name[cell] = f"UNKNOWN_{_unknown_counter:03d}"
        _unknown_counter += 1
    return _cell_unknown_name[cell]


def load_known_faces(folder):
    known = defaultdict(list)
    if not os.path.isdir(folder):
        warnings.warn(f"Known faces folder not found: {folder}")
        return {}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(
        f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts
    )
    for fname in files:
        path = os.path.join(folder, fname)
        person_name = os.path.splitext(fname)[0]
        image = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image)
        if not encs:
            warnings.warn(f"No face detected in reference image: {fname}. Skipping.")
            continue
        for enc in encs:
            known[person_name].append(enc.astype(np.float32))
    return dict(known)


def extract_keyframes(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        warnings.warn(f"Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    num = min(max_frames, total_frames)
    indices = sorted(set(np.linspace(0, total_frames - 1, num=num, dtype=int).tolist()))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    extracted = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        extracted.append((int(idx), enhanced))

    cap.release()
    return extracted


def detect_and_match(frame, known, threshold=0.6):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    detections = []

    for (top, right, bottom, left) in locations:
        h, w = frame.shape[:2]
        top = max(0, min(h, top))
        bottom = max(0, min(h, bottom))
        left = max(0, min(w, left))
        right = max(0, min(w, right))
        if bottom <= top or right <= left:
            continue

        crop = frame[top:bottom, left:right].copy()
        if crop.size == 0:
            continue

        crop_for_encoding = crop
        ch, cw = crop.shape[:2]
        if min(ch, cw) < 112:
            scale = 112.0 / float(min(ch, cw))
            nw = max(112, int(round(cw * scale)))
            nh = max(112, int(round(ch * scale)))
            crop_for_encoding = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)

        crop_rgb = cv2.cvtColor(crop_for_encoding, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(crop_rgb)
        if not encs:
            encs = face_recognition.face_encodings(rgb, known_face_locations=[(top, right, bottom, left)])
        if not encs:
            continue
        face_enc = encs[0].astype(np.float32)

        best_name = None
        best_dist = 1.0
        for name, person_encs in known.items():
            for ref in person_encs:
                dist = _cosine_distance(face_enc, ref)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        raw_matched = best_name is not None and best_dist <= threshold
        raw_label = best_name if raw_matched else "UNKNOWN"
        raw_conf = _clip_0_100((1.0 - best_dist) * 100.0) / 100.0

        cell = _grid_cell((top, right, bottom, left), frame.shape)
        _track_history[cell].append(raw_label)
        smoothed = Counter(_track_history[cell]).most_common(1)[0][0]

        if smoothed == "UNKNOWN":
            final_name = _get_unknown_for_cell(cell)
            matched = False
            confidence = 0.0
        else:
            final_name = smoothed
            matched = True
            confidence = raw_conf if raw_matched and best_name == smoothed else max(raw_conf, 0.5)

        detections.append(
            {
                "name": final_name,
                "matched": matched,
                "confidence": round(float(confidence), 4),
                "bbox": (int(top), int(right), int(bottom), int(left)),
                "face_crop": crop,
            }
        )

    return detections


def compute_face_brightness(face_crop):
    if face_crop is None or face_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return _clip_0_100(np.mean(gray) / 2.55)


def compute_eye_openness(face_crop):
    if face_crop is None or face_crop.size == 0:
        return 50.0
    try:
        mesh = _get_facemesh()
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return 50.0

        h, w = face_crop.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark
        left_ear = _ear(landmarks, LEFT_EYE_IDX, w, h)
        right_ear = _ear(landmarks, RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        score = ((avg_ear - 0.10) / (0.40 - 0.10)) * 100.0
        return _clip_0_100(score)
    except Exception:
        return 50.0


def compute_movement(prev_frame, curr_frame, bbox):
    if prev_frame is None:
        return 0.0
    top, right, bottom, left = bbox
    h, w = curr_frame.shape[:2]
    top = max(0, min(h, top))
    bottom = max(0, min(h, bottom))
    left = max(0, min(w, left))
    right = max(0, min(w, right))
    if bottom <= top or right <= left:
        return 0.0

    prev_roi = prev_frame[top:bottom, left:right]
    curr_roi = curr_frame[top:bottom, left:right]
    if prev_roi.size == 0 or curr_roi.size == 0:
        return 0.0

    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return _clip_0_100(float(np.mean(mag) * 20.0))


def aggregate_persons(all_detections, school_prefix=SCHOOL_PREFIX_DEFAULT):
    grouped = defaultdict(list)
    for det in all_detections:
        grouped[det["name"]].append(det)

    persons = []
    for name, items in grouped.items():
        brightness = float(np.mean([x["face_brightness"] for x in items])) if items else 0.0
        eye_open = float(np.mean([x["eye_openness"] for x in items])) if items else 0.0
        movement = float(np.mean([x["movement_activity"] for x in items])) if items else 0.0
        energy = _clip_0_100((brightness * 0.35) + (eye_open * 0.30) + (movement * 0.35))

        best_item = max(items, key=lambda x: _laplacian_variance(x["face_crop"]))
        matched = not name.startswith("UNKNOWN_")

        persons.append(
            {
                "name": name,
                "matched": bool(matched),
                "match_confidence": round(float(np.mean([x["confidence"] for x in items])), 4),
                "profile_image_b64": encode_b64(best_item["face_crop"]),
                "frames_detected": len(items),
                "energy_score": round(energy, 2),
                "energy_breakdown": {
                    "face_brightness": round(brightness, 2),
                    "eye_openness": round(eye_open, 2),
                    "movement_activity": round(movement, 2),
                },
                "verdict": verdict(energy),
                "first_seen_frame": int(min(x["frame_idx"] for x in items)),
                "last_seen_frame": int(max(x["frame_idx"] for x in items)),
            }
        )

    persons.sort(key=lambda p: (not p["matched"], -p["energy_score"]))
    for i, person in enumerate(persons, 1):
        person["person_id"] = f"{school_prefix}_P{i:04d}"

    return persons


def _energy_arc_svg(score):
    radius = 38
    circumference = 2 * np.pi * radius
    progress = circumference * (np.clip(score, 0.0, 100.0) / 100.0)
    dashoffset = circumference - progress
    return f"""
<svg width="110" height="110" viewBox="0 0 110 110" aria-label="Energy arc">
  <circle cx="55" cy="55" r="{radius}" stroke="#e5e7eb" stroke-width="10" fill="none"></circle>
  <circle cx="55" cy="55" r="{radius}" stroke="#2563eb" stroke-width="10" fill="none"
          stroke-linecap="round"
          transform="rotate(-90 55 55)"
          stroke-dasharray="{circumference:.2f}"
          stroke-dashoffset="{dashoffset:.2f}"></circle>
  <text x="55" y="61" text-anchor="middle" font-size="18" fill="#0f172a" font-weight="700">{score:.1f}</text>
</svg>
""".strip()


def generate_report(persons, output_path):
    total = len(persons)
    matched = sum(1 for p in persons if p["matched"])
    unknown = total - matched
    avg_energy = float(np.mean([p["energy_score"] for p in persons])) if persons else 0.0

    badge_colors = {"high": "#10b981", "moderate": "#f59e0b", "low": "#ef4444"}
    cards = []
    for p in persons:
        c = badge_colors[p["verdict"]]
        b = p["energy_breakdown"]
        cards.append(
            f"""
<article class="card">
  <div class="top">
    <img alt="{p['name']}" src="data:image/jpeg;base64,{p['profile_image_b64']}" />
    <div class="meta">
      <h2>{p['name']}</h2>
      <p>ID: {p['person_id']}</p>
      <span class="badge" style="background:{c};">{p['verdict'].upper()}</span>
      <p>Matched: {"Yes" if p["matched"] else "No"} | Confidence: {p["match_confidence"]:.2f}</p>
      <p>Frames: {p["frames_detected"]} ({p["first_seen_frame"]} - {p["last_seen_frame"]})</p>
    </div>
    <div class="arc">{_energy_arc_svg(p["energy_score"])}</div>
  </div>
  <div class="bars">
    <div><label>Brightness</label><div class="bar"><span style="width:{b["face_brightness"]:.2f}%"></span></div><small>{b["face_brightness"]:.2f}</small></div>
    <div><label>Eye Openness</label><div class="bar"><span style="width:{b["eye_openness"]:.2f}%"></span></div><small>{b["eye_openness"]:.2f}</small></div>
    <div><label>Movement</label><div class="bar"><span style="width:{b["movement_activity"]:.2f}%"></span></div><small>{b["movement_activity"]:.2f}</small></div>
  </div>
</article>
""".strip()
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sentio Mind Identity + Energy Report</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #e2e8f0;
      --bar: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; padding:28px; background:var(--bg); color:var(--text); font-family: "Segoe UI", Tahoma, sans-serif; }}
    header {{
      display:grid; grid-template-columns: repeat(4,minmax(120px,1fr)); gap:12px; margin-bottom:18px;
    }}
    .stat {{
      background: var(--card); border:1px solid var(--line); border-radius:12px; padding:12px;
    }}
    .stat h3 {{ margin:0 0 6px 0; color:var(--muted); font-size:13px; font-weight:600; }}
    .stat p {{ margin:0; font-size:22px; font-weight:700; }}
    main {{ display:grid; gap:14px; }}
    .card {{
      background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px;
    }}
    .top {{ display:grid; grid-template-columns: 120px 1fr 130px; gap:14px; align-items:center; }}
    img {{ width:120px; height:120px; object-fit:cover; border-radius:10px; border:1px solid var(--line); }}
    h2 {{ margin:0 0 6px 0; font-size:20px; }}
    .meta p {{ margin:3px 0; color:var(--muted); font-size:13px; }}
    .badge {{ display:inline-block; color:#fff; font-size:12px; font-weight:700; border-radius:999px; padding:4px 10px; margin-bottom:6px; }}
    .bars {{ margin-top:10px; display:grid; gap:8px; }}
    .bars label {{ display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }}
    .bar {{ height:9px; border-radius:999px; background:#e2e8f0; overflow:hidden; }}
    .bar span {{ display:block; height:100%; background:var(--bar); }}
    small {{ color:var(--muted); font-size:11px; }}
    @media (max-width: 900px) {{
      header {{ grid-template-columns: repeat(2,minmax(120px,1fr)); }}
      .top {{ grid-template-columns: 90px 1fr; }}
      .arc {{ grid-column: span 2; }}
      img {{ width:90px; height:90px; }}
    }}
  </style>
</head>
<body>
  <header>
    <section class="stat"><h3>Total Persons</h3><p>{total}</p></section>
    <section class="stat"><h3>Matched</h3><p>{matched}</p></section>
    <section class="stat"><h3>Unknown</h3><p>{unknown}</p></section>
    <section class="stat"><h3>Average Energy</h3><p>{avg_energy:.2f}</p></section>
  </header>
  <main>
    {"".join(cards) if cards else "<p>No faces detected.</p>"}
  </main>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def write_integration_json(persons, output_path, video_name, proc_time):
    matched = sum(1 for p in persons if p["matched"])
    unknown = len(persons) - matched

    payload = {
        "source": "sentio-poc-identity-energy",
        "school": SCHOOL_NAME,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "video_file": video_name,
        "total_persons_matched": matched,
        "total_persons_unknown": unknown,
        "processing_time_sec": round(float(proc_time), 3),
        "persons": [],
    }

    for p in persons:
        payload["persons"].append(
            {
                "person_id": p["person_id"],
                "name": p["name"],
                "matched": p["matched"],
                "match_confidence": p["match_confidence"],
                "profile_image_b64": p["profile_image_b64"],
                "frames_detected": p["frames_detected"],
                "energy_score": p["energy_score"],
                "energy_breakdown": {
                    "face_brightness": p["energy_breakdown"]["face_brightness"],
                    "eye_openness": p["energy_breakdown"]["eye_openness"],
                    "movement_activity": p["energy_breakdown"]["movement_activity"],
                },
                "verdict": p["verdict"],
                "first_seen_frame": p["first_seen_frame"],
                "last_seen_frame": p["last_seen_frame"],
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _annotate_frame(frame, detections):
    out = frame.copy()
    for det in detections:
        top, right, bottom, left = det["bbox"]
        color = (16, 185, 129) if det["matched"] else (68, 68, 239)
        cv2.rectangle(out, (left, top), (right, bottom), color, 2)
        label = f"{det['name']} {det['confidence']:.2f}"
        y = max(18, top - 8)
        cv2.putText(
            out,
            label,
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _write_demo_video(annotated_frames, output_path, fps=5.0):
    if not annotated_frames:
        return False

    first = annotated_frames[0][1]
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        return False

    for _, frame in annotated_frames:
        writer.write(frame)
    writer.release()
    return True


def _print_summary_table(persons):
    print("\nFinal Summary")
    print("=" * 94)
    print(
        f"{'Person ID':<12} {'Name':<22} {'Matched':<8} {'Energy':<8} "
        f"{'Frames':<7} {'Verdict':<10} {'Confidence':<10}"
    )
    print("-" * 94)
    for p in persons:
        print(
            f"{p['person_id']:<12} {p['name']:<22} {str(p['matched']):<8} "
            f"{p['energy_score']:<8.2f} {p['frames_detected']:<7d} "
            f"{p['verdict']:<10} {p['match_confidence']:<10.2f}"
        )
    print("=" * 94)


def main():
    parser = argparse.ArgumentParser(
        description="Sentio Mind face recognition + energy scoring pipeline"
    )
    parser.add_argument("--known_faces", default="known_faces", help="Known faces folder")
    parser.add_argument("--video", default="data/video_sample_1.mov", help="Input video path")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=20, help="Maximum keyframes")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine distance threshold")
    parser.add_argument(
        "--school_prefix",
        type=str,
        default=SCHOOL_PREFIX_DEFAULT,
        help="Prefix for person_id",
    )
    args = parser.parse_args()

    global _unknown_counter
    _track_history.clear()
    _cell_unknown_name.clear()
    _unknown_counter = 1

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/7] Loading known faces...")
    known = load_known_faces(args.known_faces)
    total_known_encodings = sum(len(v) for v in known.values())
    print(
        f"      Loaded {len(known)} identities with {total_known_encodings} total embeddings."
    )

    print("[2/7] Extracting keyframes with CLAHE on LAB L-channel...")
    keyframes = extract_keyframes(args.video, max_frames=args.max_frames)
    print(f"      Extracted {len(keyframes)} keyframes.")

    print("[3/7] Detecting faces, matching identities, and temporal smoothing...")
    all_detections = []
    annotated = []
    prev_frame = None
    for i, (frame_idx, frame) in enumerate(keyframes, 1):
        detections = detect_and_match(frame, known, threshold=args.threshold)
        for det in detections:
            det["frame_idx"] = int(frame_idx)
            det["face_brightness"] = compute_face_brightness(det["face_crop"])
            det["eye_openness"] = compute_eye_openness(det["face_crop"])
            det["movement_activity"] = compute_movement(prev_frame, frame, det["bbox"])
            all_detections.append(det)
        annotated.append((frame_idx, _annotate_frame(frame, detections)))
        prev_frame = frame
        print(f"      Frame {i}/{len(keyframes)} (idx={frame_idx}): {len(detections)} faces")

    print("[4/7] Aggregating per-person energy components...")
    persons = aggregate_persons(all_detections, school_prefix=args.school_prefix)
    print(f"      Aggregated into {len(persons)} unique persons.")

    print("[5/7] Generating offline HTML report...")
    report_path = os.path.join(args.output_dir, "report.html")
    generate_report(persons, report_path)
    print(f"      Wrote report: {report_path}")

    print("[6/7] Writing integration JSON...")
    json_path = os.path.join(args.output_dir, "integration_output.json")
    processing_time = time.time() - t0
    write_integration_json(persons, json_path, os.path.basename(args.video), processing_time)
    print(f"      Wrote integration JSON: {json_path}")

    print("[7/7] Building demo video...")
    demo_path = os.path.join(args.output_dir, "demo.mp4")
    demo_ok = _write_demo_video(annotated, demo_path, fps=5.0)
    if demo_ok:
        print(f"      Wrote demo video: {demo_path}")
    else:
        print("      Could not write demo video (no frames or writer init failed).")

    _print_summary_table(persons)
    print(f"Total detections processed: {len(all_detections)}")
    print(f"Processing time: {processing_time:.3f} sec")


if __name__ == "__main__":
    main()
