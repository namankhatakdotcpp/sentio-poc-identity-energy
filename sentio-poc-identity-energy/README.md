# Sentio Identity + Energy Platform

A production-ready computer vision system for identity recognition and engagement-energy scoring from classroom/CCTV video, with deployable API infrastructure on Render.

This repository includes:
- ML pipeline (batch + realtime)
- Hybrid tracking and identity smoothing
- Offline analytics artifacts (JSON, CSV, HTML)
- FastAPI service for job-based video processing
- Docker + Render deployment configuration
- Automated API tests

---

## 1. Project Overview

Sentio is designed as a practical ML systems project, not a notebook demo. It solves a real workflow:

1. ingest known identities and video footage
2. detect faces robustly (including low-resolution crops)
3. stabilize identity over time using hybrid association and confidence decay
4. compute interpretable energy signals per person
5. publish consumable outputs for dashboards and downstream systems

Outputs are generated as:
- `integration_output.json` (strict schema for integrations)
- `integration_output.csv` (analytics-friendly export)
- `report.html` (offline visual report with timeline)
- `demo.mp4` (optional annotated playback)

---

## 2. Feature Set

### ML / Vision
- face encoding from `known_faces/`
- MTCNN detection with bicubic upscale for tiny faces
- hybrid tracking association (IoU + center distance + embedding similarity)
- temporal identity smoothing with track confidence decay/boost
- UNKNOWN identity management

### Energy Scoring
- brightness signal
- eye openness via MediaPipe FaceMesh EAR
- motion via Farneback optical flow
- weighted score aggregation per person

### System / Platform
- centralized config with env + JSON overrides
- structured logging events
- batch and realtime processing modes
- FastAPI endpoint for video processing jobs
- upload safety checks (type, size, timeout)
- Dockerized deployment and Render blueprint

---

## 3. Architecture

```text
known_faces/ + video
   ↓
encoder.py  -> known embeddings
   ↓
detector.py -> face boxes + crops
   ↓
matcher.py  -> hybrid tracking + identity smoothing
   ↓
energy.py   -> brightness + eye openness + motion
   ↓
aggregator.py -> per-person aggregation + best profile image
   ↓
reporter.py -> JSON + CSV + offline HTML report + timeline
   ↓
solution.py / api.py
```

`api.py` wraps `solution.py` pipeline execution in an HTTP service.

---

## 4. Repository Structure

```text
sentio-poc-identity-energy/
├── api.py
├── known_faces/
├── data/
│   └── video_sample_1.mov
├── src/
│   ├── config.py
│   ├── encoder.py
│   ├── detector.py
│   ├── matcher.py
│   ├── energy.py
│   ├── aggregator.py
│   └── reporter.py
├── tests/
│   └── test_api.py
├── outputs/
│   ├── report.html
│   ├── integration_output.json
│   ├── integration_output.csv
│   └── profile_crops/
├── config.example.json
├── Dockerfile
├── render.yaml
├── solution.py
├── requirements.txt
└── README.md
```

---

## 5. Local Development

### Prerequisites
- Python 3.10+
- ffmpeg and OpenCV runtime libs available on your machine

### Install dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart pytest
```

### Run batch pipeline

```bash
python solution.py \
  --known_faces known_faces \
  --video data/video_sample_1.mov \
  --output_dir outputs
```

### Run realtime pipeline

```bash
python solution.py \
  --known_faces known_faces \
  --video 0 \
  --realtime \
  --output_dir outputs
```

---

## 6. FastAPI Usage

### Start API locally

```bash
uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Health check

```bash
curl http://localhost:8000/health
```

### Process a video

```bash
curl -X POST "http://localhost:8000/process-video/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/video_sample_1.mov"
```

Example response:

```json
{
  "status": "completed",
  "job_id": "8e5f7e6f4cd2478dbf6d80e3008f89d8",
  "report_path": ".../outputs/jobs/<job_id>/outputs/report.html",
  "json_path": ".../outputs/jobs/<job_id>/outputs/integration_output.json",
  "csv_path": ".../outputs/jobs/<job_id>/outputs/integration_output.csv",
  "processing_time_sec": 14.273
}
```

---

## 7. Configuration and Environment

### Environment variables
- `PORT` (default: `8000`)
- `LOG_LEVEL` (default: `INFO`)
- `MAX_UPLOAD_MB` (default: `200`)
- `PROCESS_TIMEOUT_SEC` (default: `1200`)

### Config overrides
Use `config.example.json` as a template and pass it via pipeline CLI:

```bash
python solution.py --config my_config.json
```

`src/config.py` merges defaults + JSON config + environment overrides.

---

## 8. Deploy on Render

### Option A: Blueprint deploy (`render.yaml`)
1. Push this repo to GitHub.
2. In Render, create a new Blueprint service from the repo.
3. Render reads `render.yaml` and provisions the Docker web service.

### Option B: Manual Render setup
1. Create a new Web Service.
2. Choose Docker environment.
3. Set env vars:
   - `PORT=8000`
   - `LOG_LEVEL=INFO`
4. Deploy.

Docker runtime command:

```bash
uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

---

## 9. Testing

Run API tests:

```bash
pytest tests/test_api.py -q
```

Current tests validate:
- success response contract for `/process-video/`
- invalid file rejection
- output schema keys in API response

---

## 10. Demo Flow

1. Place identity images in `known_faces/`
2. Send video through API or run batch script
3. Open generated `report.html`
4. Inspect timeline, confidence bars, CSV export, and strict JSON output

This setup is designed for internship-to-production ML systems portfolios: model logic + data products + deployment discipline.
