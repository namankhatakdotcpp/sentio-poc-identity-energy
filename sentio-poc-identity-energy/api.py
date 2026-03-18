from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile


APP_ROOT = Path(__file__).resolve().parent
JOBS_ROOT = APP_ROOT / "outputs" / "jobs"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
PROCESS_TIMEOUT_SEC = int(os.getenv("PROCESS_TIMEOUT_SEC", "1200"))


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("sentio.api")
    if not logger.handlers:
        level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


logger = _setup_logging()
app = FastAPI(title="Sentio Identity + Energy API", version="1.0.0")


@app.on_event("startup")
def warmup_model() -> None:
    """Pre-warm embedding model on startup to eliminate first-request lag."""
    try:
        from src.embedding_model import get_embedding_safe

        logger.info("Warming up embedding model...")
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        get_embedding_safe(dummy)
        logger.info("Embedding model warmup complete")
    except Exception as e:
        logger.warning(f"Model warmup failed (non-fatal): {str(e)}")


def _event(name: str, **payload: Any) -> None:
    logger.info(json.dumps({"event": name, **payload}, default=str))


def _validate_upload(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{ext}'. "
                f"Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
            ),
        )


async def _save_upload(file: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with destination.open("wb") as out:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_MB} MB upload limit.",
                )
            out.write(chunk)

    return written


def _run_pipeline_job(video_path: str, output_dir: str, known_faces_dir: str) -> Dict[str, Any]:
    # Lazy import so tests can mock this function without requiring heavy CV deps.
    from solution import _setup_logger, run_batch_pipeline  # type: ignore
    from src.config import load_config  # type: ignore

    cfg = load_config()
    cfg.runtime.realtime = False
    cfg.runtime.no_display = True

    pipeline_logger = _setup_logger(cfg)
    return run_batch_pipeline(
        known_faces_dir=known_faces_dir,
        video_path=video_path,
        output_dir=output_dir,
        cfg=cfg,
        logger=pipeline_logger,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    started = time.time()
    job_id = uuid.uuid4().hex

    _event("request_received", job_id=job_id, filename=file.filename)
    logger.info(f"[{job_id}] Starting video upload: {file.filename}")

    _validate_upload(file)

    job_dir = JOBS_ROOT / job_id
    upload_dir = job_dir / "input"
    output_dir = job_dir / "outputs"
    known_faces_dir = str(APP_ROOT / "known_faces")

    upload_path = upload_dir / (file.filename or "input_video.mp4")

    try:
        upload_start = time.time()
        size_bytes = await _save_upload(file, upload_path)
        upload_time = time.time() - upload_start
        logger.info(f"[{job_id}] Upload complete: {size_bytes} bytes in {upload_time:.2f}s")

        process_start = time.time()
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _run_pipeline_job,
                str(upload_path),
                str(output_dir),
                known_faces_dir,
            ),
            timeout=PROCESS_TIMEOUT_SEC,
        )
        process_time = time.time() - process_start
        logger.info(f"[{job_id}] Processing complete in {process_time:.2f}s")

        runtime_sec = round(time.time() - started, 3)
        _event(
            "request_completed",
            job_id=job_id,
            processing_time_sec=runtime_sec,
            uploaded_bytes=size_bytes,
        )
        logger.info(f"[{job_id}] Total request time: {runtime_sec}s")

        return {
            "status": "completed",
            "job_id": job_id,
            "report_path": result["report_path"],
            "json_path": result["json_path"],
            "csv_path": result["csv_path"],
            "processing_time_sec": runtime_sec,
        }
    except asyncio.TimeoutError as exc:
        runtime_sec = round(time.time() - started, 3)
        _event("request_timeout", job_id=job_id, processing_time_sec=runtime_sec)
        raise HTTPException(
            status_code=504,
            detail=f"Processing timed out after {PROCESS_TIMEOUT_SEC} seconds.",
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        runtime_sec = round(time.time() - started, 3)
        logger.exception(f"[{job_id}] Pipeline failed", exc_info=True)
        _event("request_failed", job_id=job_id, processing_time_sec=runtime_sec, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
