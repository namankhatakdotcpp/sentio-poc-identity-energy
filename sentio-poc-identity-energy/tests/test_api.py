from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_process_video_success(monkeypatch):
    def fake_run(video_path: str, output_dir: str, known_faces_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {
            "report_path": str(out / "report.html"),
            "json_path": str(out / "integration_output.json"),
            "csv_path": str(out / "integration_output.csv"),
            "persons": [],
            "demo_path": str(out / "demo.mp4"),
            "demo_ok": True,
        }

    monkeypatch.setattr(api, "_run_pipeline_job", fake_run)

    files = {"file": ("sample.mp4", BytesIO(b"video-bytes"), "video/mp4")}
    response = client.post("/process-video/", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "job_id" in body
    assert body["report_path"].endswith("report.html")
    assert body["json_path"].endswith("integration_output.json")
    assert body["csv_path"].endswith("integration_output.csv")
    assert "processing_time_sec" in body


def test_process_video_rejects_invalid_extension():
    files = {"file": ("not_video.txt", BytesIO(b"hello"), "text/plain")}
    response = client.post("/process-video/", files=files)

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_process_video_schema_keys(monkeypatch):
    def fake_run(video_path: str, output_dir: str, known_faces_dir: str):
        out = Path(output_dir)
        return {
            "report_path": str(out / "report.html"),
            "json_path": str(out / "integration_output.json"),
            "csv_path": str(out / "integration_output.csv"),
            "persons": [{"id": "SM_P0001"}],
            "demo_path": str(out / "demo.mp4"),
            "demo_ok": False,
        }

    monkeypatch.setattr(api, "_run_pipeline_job", fake_run)

    files = {"file": ("clip.mov", BytesIO(b"1234"), "video/quicktime")}
    response = client.post("/process-video/", files=files)

    assert response.status_code == 200
    body = response.json()

    expected_keys = {
        "status",
        "job_id",
        "report_path",
        "json_path",
        "csv_path",
        "processing_time_sec",
    }
    assert expected_keys.issubset(set(body.keys()))
