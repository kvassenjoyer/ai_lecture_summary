"""Opt-in tests against the actual Compose PostgreSQL and S3 service."""
import io
import os
import wave
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from lecture_service.config import Settings
from lecture_service.db import LectureJob
from lecture_service.main import create_app
from lecture_service.storage import create_storage
from lecture_service.worker import Worker

pytestmark = pytest.mark.skipif(os.environ.get("RUN_INFRA_TESTS") != "1", reason="requires Compose infrastructure")


def settings_for(tmp_path):
    return Settings(database_url="postgresql+psycopg://lecture:lecture_local@127.0.0.1:5433/lectures",
                    storage_backend="s3", s3_endpoint_url="http://127.0.0.1:9000",
                    s3_access_key_id="lecture_minio", s3_secret_access_key="lecture_minio_local_secret",
                    data_dir=tmp_path, _env_file=None)


def test_live_s3_binary_roundtrip(tmp_path):
    storage = create_storage(settings_for(tmp_path))
    source = tmp_path / "binary.wav"
    source.write_bytes(bytes(range(256)) * 1000)
    key = f"integration/{uuid4()}/audio.wav"
    try:
        storage.put_file(key, source)
        with storage.materialize(key) as restored:
            assert restored.read_bytes() == source.read_bytes()
        assert not restored.exists()
    finally:
        storage.delete(key)


def test_live_postgres_s3_job_survives_app_restart(tmp_path):
    settings = settings_for(tmp_path)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    payload = buffer.getvalue()

    class ControlledPipeline:
        def process(self, path, language, progress):
            assert path.read_bytes() == payload
            assert language == "ru"
            progress("transcribing", 0.5)
            return {"transcript": "Проверка хранения.", "summary": "Проверка сохранена.",
                    "raw_transcript": "Проверка хранения.", "segments": [], "duration_seconds": 0.1,
                    "metadata": {"controlled_pipeline": True}}

    job_id = None
    storage = create_storage(settings)
    try:
        with TestClient(create_app(settings)) as client:
            job = client.post("/api/lectures", files={"file": ("infra.wav", payload, "audio/wav")},
                              data={"language": "ru"}).raise_for_status().json()
            job_id = job["id"]
            # Run only our job; no background worker should be active during this diagnostic.
            worker = Worker(settings, pipeline=ControlledPipeline(), session_factory=client.app.state.sessions,
                            storage=client.app.state.storage)
            assert worker.run_once()
            assert client.get(f"/api/lectures/{job_id}").json()["status"] == "completed"
        with TestClient(create_app(settings)) as restarted:
            assert restarted.get("/health").json()["database_backend"] == "postgresql"
            result = restarted.get(f"/api/lectures/{job_id}/result").raise_for_status().json()
            assert result["transcript"] == "Проверка хранения."
            with restarted.app.state.sessions.begin() as session:
                session.execute(delete(LectureJob).where(LectureJob.id == job_id))
    finally:
        if job_id:
            storage.delete(f"{job_id}/audio.wav")
