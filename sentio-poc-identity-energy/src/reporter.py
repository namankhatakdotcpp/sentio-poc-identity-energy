from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from html import escape
from typing import Dict, List, Optional


def _arc_svg(score: float) -> str:
    radius = 36
    circumference = 2.0 * 3.14159265 * radius
    pct = max(0.0, min(100.0, score)) / 100.0
    offset = circumference * (1.0 - pct)
    return f"""
<svg width="110" height="110" viewBox="0 0 110 110" role="img" aria-label="Energy score">
  <circle cx="55" cy="55" r="{radius}" fill="none" stroke="#d1d5db" stroke-width="10"></circle>
  <circle cx="55" cy="55" r="{radius}" fill="none" stroke="#2563eb" stroke-width="10" stroke-linecap="round"
    transform="rotate(-90 55 55)" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"></circle>
  <text x="55" y="61" font-size="18" text-anchor="middle" fill="#111827" font-weight="700">{score:.1f}</text>
</svg>
""".strip()


def _badge_color(verdict: str) -> str:
    verdict_upper = (verdict or "").upper()
    if verdict_upper == "HIGH":
        return "#10b981"
    if verdict_upper == "MEDIUM":
        return "#f59e0b"
    return "#ef4444"


def _pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def write_json(persons: List[Dict], output_path: str) -> None:
    payload = []
    for p in persons:
        payload.append(
            {
                "id": p["id"],
                "name": p["name"],
                "energy_score": p["energy_score"],
                "brightness": p["brightness"],
                "eye_openness": p["eye_openness"],
                "motion": p["motion"],
                "frames": p["frames"],
                "time_range": p["time_range"],
                "profile_image_base64": p["profile_image_base64"],
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(persons: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    columns = [
        "id",
        "name",
        "energy_score",
        "brightness",
        "eye_openness",
        "motion",
        "frames",
        "time_range_start",
        "time_range_end",
        "avg_match_confidence",
        "avg_track_confidence",
        "verdict",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for person in persons:
            writer.writerow(
                {
                    "id": person["id"],
                    "name": person["name"],
                    "energy_score": person["energy_score"],
                    "brightness": person["brightness"],
                    "eye_openness": person["eye_openness"],
                    "motion": person["motion"],
                    "frames": person["frames"],
                    "time_range_start": person["time_range"][0],
                    "time_range_end": person["time_range"][1],
                    "avg_match_confidence": person.get("avg_match_confidence", 0.0),
                    "avg_track_confidence": person.get("avg_track_confidence", 0.0),
                    "verdict": person.get("verdict", "LOW"),
                }
            )


def _timeline_svg(persons: List[Dict], timeline_records: Optional[List[Dict]]) -> str:
    if not persons:
        return ""

    if timeline_records:
        person_frames = defaultdict(list)
        max_frame = 0
        for row in timeline_records:
            person_frames[row["name"]].append(int(row["frame_idx"]))
            max_frame = max(max_frame, int(row["frame_idx"]))
    else:
        person_frames = defaultdict(list)
        max_frame = max(p["time_range"][1] for p in persons)
        for person in persons:
            person_frames[person["name"]] = [person["time_range"][0], person["time_range"][1]]

    max_frame = max(max_frame, 1)
    width = 980
    left_pad = 220
    timeline_w = 720
    row_h = 34
    top_pad = 24
    height = top_pad + (len(persons) * row_h) + 24

    rows = []
    rows.append(f'<line x1="{left_pad}" y1="12" x2="{left_pad + timeline_w}" y2="12" stroke="#cbd5e1" stroke-width="1"/>')

    for idx, person in enumerate(persons):
        y = top_pad + (idx * row_h)
        name = escape(person["name"])
        rows.append(f'<text x="8" y="{y + 5}" fill="#334155" font-size="12">{name}</text>')

        rows.append(
            f'<line x1="{left_pad}" y1="{y}" x2="{left_pad + timeline_w}" y2="{y}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )

        frames = sorted(person_frames.get(person["name"], []))
        if not frames:
            continue

        start = frames[0]
        end = frames[-1]
        start_x = left_pad + int((start / max_frame) * timeline_w)
        end_x = left_pad + int((end / max_frame) * timeline_w)
        rows.append(
            f'<line x1="{start_x}" y1="{y}" x2="{max(start_x + 2, end_x)}" y2="{y}" '
            'stroke="#2563eb" stroke-width="4" stroke-linecap="round"/>'
        )

        sampled = frames[:: max(1, len(frames) // 16)]
        for frame_idx in sampled:
            px = left_pad + int((frame_idx / max_frame) * timeline_w)
            rows.append(f'<circle cx="{px}" cy="{y}" r="2" fill="#0f172a"/>')

    rows.append(f'<text x="{left_pad}" y="{height - 6}" fill="#64748b" font-size="11">frame 0</text>')
    rows.append(
        f'<text x="{left_pad + timeline_w - 80}" y="{height - 6}" fill="#64748b" font-size="11">frame {max_frame}</text>'
    )

    return f"""
<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Identity timeline">
  {''.join(rows)}
</svg>
""".strip()


def _render_person_card(person: Dict) -> str:
    name = escape(str(person["name"]))
    person_id = escape(str(person["id"]))
    badge = person.get("verdict", "LOW").upper()
    badge_color = _badge_color(badge)

    brightness = _pct(person["brightness"])
    eye = _pct(person["eye_openness"])
    motion = _pct(person["motion"])
    match_conf = _pct(person.get("avg_match_confidence", 0.0) * 100.0)
    track_conf = _pct(person.get("avg_track_confidence", 0.0) * 100.0)

    return f"""
<article class="card">
  <div class="card-head">
    <img src="data:image/jpeg;base64,{person['profile_image_base64']}" alt="{name} profile" />
    <div class="identity">
      <h3>{name}</h3>
      <p class="meta">{person_id} | frames: {person['frames']} | range: {person['time_range'][0]}-{person['time_range'][1]}</p>
      <span class="badge" style="background:{badge_color};">{badge}</span>
      <p class="meta">match conf: {person.get('avg_match_confidence', 0.0):.2f} | track conf: {person.get('avg_track_confidence', 0.0):.2f}</p>
    </div>
    <div class="arc-wrap">{_arc_svg(person['energy_score'])}</div>
  </div>
  <div class="bars">
    <div class="bar-item"><label>Brightness</label><div class="bar"><span style="width:{brightness:.2f}%"></span></div><small>{person['brightness']:.2f}</small></div>
    <div class="bar-item"><label>Eye Openness</label><div class="bar"><span style="width:{eye:.2f}%"></span></div><small>{person['eye_openness']:.2f}</small></div>
    <div class="bar-item"><label>Motion</label><div class="bar"><span style="width:{motion:.2f}%"></span></div><small>{person['motion']:.2f}</small></div>
    <div class="bar-item"><label>Match Confidence</label><div class="bar"><span style="width:{match_conf:.2f}%"></span></div><small>{person.get('avg_match_confidence', 0.0):.2f}</small></div>
    <div class="bar-item"><label>Track Confidence</label><div class="bar"><span style="width:{track_conf:.2f}%"></span></div><small>{person.get('avg_track_confidence', 0.0):.2f}</small></div>
  </div>
</article>
""".strip()


def write_html_report(persons: List[Dict], output_path: str, timeline_records: Optional[List[Dict]] = None) -> None:
    total = len(persons)
    matched = sum(1 for p in persons if not p["name"].startswith("UNKNOWN_"))
    unknown = total - matched
    avg_energy = sum(p["energy_score"] for p in persons) / total if total else 0.0
    cards_html = "".join(_render_person_card(person) for person in persons)
    timeline_html = _timeline_svg(persons, timeline_records)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sentio Identity + Energy Report</title>
  <style>
    :root {{
      --bg: #f3f4f6;
      --card: #ffffff;
      --line: #e5e7eb;
      --text: #111827;
      --muted: #6b7280;
      --bar: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    h1 {{ margin: 0 0 14px 0; font-size: 26px; }}
    h2 {{ margin: 0 0 10px 0; font-size: 20px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    .stat .k {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .stat .v {{ font-size: 22px; font-weight: 700; }}
    .timeline {{
      margin-bottom: 18px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .grid {{ display: grid; gap: 14px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .card-head {{
      display: grid;
      grid-template-columns: 110px 1fr 120px;
      gap: 12px;
      align-items: center;
    }}
    img {{
      width: 110px;
      height: 110px;
      border-radius: 10px;
      object-fit: cover;
      border: 1px solid var(--line);
    }}
    h3 {{ margin: 0 0 6px 0; font-size: 20px; }}
    .meta {{ margin: 0; color: var(--muted); font-size: 12px; }}
    .badge {{
      display: inline-block;
      margin-top: 8px;
      color: white;
      font-size: 11px;
      font-weight: 700;
      border-radius: 999px;
      padding: 4px 10px;
    }}
    .bars {{ margin-top: 12px; display: grid; gap: 8px; }}
    .bar-item label {{ font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }}
    .bar {{ height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: var(--bar); }}
    small {{ font-size: 11px; color: var(--muted); }}
    @media (max-width: 900px) {{
      .summary {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .card-head {{ grid-template-columns: 90px 1fr; }}
      .arc-wrap {{ grid-column: span 2; }}
      img {{ width: 90px; height: 90px; }}
    }}
  </style>
</head>
<body>
  <h1>Sentio POC: Identity + Energy Report</h1>
  <section class="summary">
    <div class="stat"><div class="k">Total Persons</div><div class="v">{total}</div></div>
    <div class="stat"><div class="k">Matched</div><div class="v">{matched}</div></div>
    <div class="stat"><div class="k">Unknown</div><div class="v">{unknown}</div></div>
    <div class="stat"><div class="k">Avg Energy</div><div class="v">{avg_energy:.2f}</div></div>
  </section>
  <section class="timeline">
    <h2>Identity Timeline</h2>
    {timeline_html if timeline_html else "<p>No timeline data available.</p>"}
  </section>
  <section class="grid">
    {cards_html if cards_html else "<p>No detections available.</p>"}
  </section>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
