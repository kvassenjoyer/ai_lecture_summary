import io
import wave

import pytest
from fastapi.testclient import TestClient

from lecture_service.config import Settings
from lecture_service.main import create_app
from lecture_service.worker import Worker


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", data_dir=tmp_path,
                        max_upload_mb=0.1, _env_file=None)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_upload_persists_safe_filename_and_queued_status(client):
    response = client.post("/api/lectures", files={"file": ("../../lecture.wav", wav_bytes(), "audio/wav")},
                           data={"language": "ru"})
    assert response.status_code == 202
    job = response.json()
    assert job["filename"] == "lecture.wav"
    assert job["status"] == "queued"
    assert job["language"] == "ru"
    assert job["created_at"].endswith("+00:00")
    assert "object_key" not in job
    assert client.get(f"/api/lectures/{job['id']}").json()["id"] == job["id"]
    listing = client.get("/api/lectures").json()
    assert listing["total"] == 1
    assert listing["items"][0]["id"] == job["id"]
    assert client.get(f"/api/lectures/{job['id']}/result").status_code == 409
    assert len(list((client.app.state.settings.data_dir / "uploads").rglob("*.wav"))) == 1
    assert not list((client.app.state.settings.data_dir / "tmp").iterdir())


@pytest.mark.parametrize("name,content,status", [
    ("payload.exe", b"MZ", 415),
    ("fake.wav", b"not audio", 415),
    ("empty.wav", b"", 400),
    ("big.wav", wav_bytes() + b"\0" * 200_000, 413),
], ids=["extension", "signature", "empty", "oversized"])
def test_reject_bad_uploads_and_cleanup(client, name, content, status):
    response = client.post("/api/lectures", files={"file": (name, content, "audio/wav")})
    assert response.status_code == status
    assert client.get("/api/lectures").json()["total"] == 0
    assert not list((client.app.state.settings.data_dir / "uploads").rglob("*.wav"))
    temporary_dir = client.app.state.settings.data_dir / "tmp"
    assert not temporary_dir.exists() or not list(temporary_dir.iterdir())


def test_invalid_language_and_missing_job(client):
    response = client.post("/api/lectures", files={"file": ("lecture.wav", wav_bytes(), "audio/wav")},
                           data={"language": "ru; rm -rf"})
    assert response.status_code == 422
    assert client.get("/api/lectures/does-not-exist").status_code == 404
    assert client.get("/api/lectures?limit=1000").status_code == 422


def test_health_and_result_downloads(client):
    class Pipeline:
        def process(self, audio_path, language, progress):
            assert audio_path.is_file()
            progress("transcribing", 0.4)
            return {"transcript": "Текст лекции.", "raw_transcript": "Текст лекции.",
                    "summary": "- Главная мысль.", "segments": [{"start": 0, "end": 0.1, "text": "Текст лекции."}],
                    "duration_seconds": 0.1, "metadata": {"test": True}}

    job = client.post("/api/lectures", files={"file": ("lecture.wav", wav_bytes(), "audio/wav")}).json()
    worker = Worker(client.app.state.settings, pipeline=Pipeline(), session_factory=client.app.state.sessions,
                    storage=client.app.state.storage)
    assert worker.run_once()
    final_job = client.get(f"/api/lectures/{job['id']}").json()
    assert final_job["status"] == "completed" and final_job["progress"] == 1
    result = client.get(f"/api/lectures/{job['id']}/result").json()
    assert result["transcript"] == "Текст лекции."
    for extension in ("md", "txt", "json"):
        response = client.get(f"/api/lectures/{job['id']}/download?format={extension}")
        assert response.status_code == 200
        assert "Главная мысль" in response.text
        assert f".{extension}" in response.headers["content-disposition"]
    assert "## Конспект" in client.get(f"/api/lectures/{job['id']}/download").text
    assert client.get(f"/api/lectures/{job['id']}/download?format=exe").status_code == 422
    assert client.get("/health").json()["database_backend"] == "sqlite"


def test_body_limit_also_rejects_chunked_requests(client):
    # No Content-Length: count receive chunks, before multipart has spooled the file.
    boundary = "test-boundary"
    def chunks():
        yield f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n\r\n'.encode()
        yield b"RIFF" + b"\0" * 500_000
        yield f"\r\n--{boundary}--\r\n".encode()
    response = client.post("/api/lectures", content=chunks(),
                           headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413
