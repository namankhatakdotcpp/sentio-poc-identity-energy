# sentio-poc-identity-energy

Production-style POC pipeline for identity recognition and per-person energy scoring from classroom/CCTV video.

## Features

- Reference encoding from `known_faces/` using `face_recognition`
- Face detection with `MTCNN`
- Low-resolution face handling (bicubic upscale for faces below 112px)
- Identity matching using cosine similarity
- Temporal identity smoothing (rolling window of last 5 frames)
- Unknown labeling (`UNKNOWN_001`, `UNKNOWN_002`, ...)
- Energy scoring from:
  - Brightness
  - Eye openness (MediaPipe FaceMesh EAR)
  - Motion (Farneback optical flow)
- Multi-frame aggregation per person
- Sharpest profile crop selection via Laplacian variance
- Offline HTML report generation
- JSON integration output with strict schema
- Optional annotated `demo.mp4`

## Repository Layout

```text
sentio-poc-identity-energy/
├── known_faces/
├── data/
│   └── video_sample_1.mov
├── src/
│   ├── encoder.py
│   ├── detector.py
│   ├── matcher.py
│   ├── energy.py
│   ├── aggregator.py
│   └── reporter.py
├── outputs/
│   ├── report.html
│   ├── integration_output.json
│   └── profile_crops/
├── solution.py
├── requirements.txt
└── README.md
```

## Setup

1. Use Python 3.10+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put one image per known person in `known_faces/`.
4. Put input video at `data/video_sample_1.mov`.

## Run

```bash
python solution.py \
  --known_faces known_faces \
  --video data/video_sample_1.mov \
  --output_dir outputs
```

## Outputs

- `outputs/report.html` (offline report)
- `outputs/integration_output.json` (strict integration schema)
- `outputs/demo.mp4` (annotated preview; optional)
- `outputs/profile_crops/*.jpg` (best profile per identity)

## JSON Output Schema

```json
[
  {
    "id": "SM_P0001",
    "name": "John Doe",
    "energy_score": 61.25,
    "brightness": 58.20,
    "eye_openness": 70.10,
    "motion": 55.80,
    "frames": 18,
    "time_range": [35, 214],
    "profile_image_base64": "..."
  }
]
```
