import io
import wave
from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import inspect, select

from lecture_service.config import Settings
from lecture_service.db import LectureJob, create_database, utcnow
from lecture_service.main import create_app
from lecture_service.worker import Worker
from lecture_service.youtube import YouTubeDownloadError

VIDEO_URL = "https://www.youtube.com/watch?v=BaW_jenozKc"


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    return buffer.getvalue()


def test_youtube_route_download_storage_pipeline_and_provenance(tmp_path, monkeypatch):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'jobs.db'}", data_dir=tmp_path, _env_file=None)
    calls = []

    def download(url, directory, settings, progress):
        assert url == VIDEO_URL
        calls.append(url)
        path = directory / "BaW_jenozKc.wav"
        path.write_bytes(wav_bytes())
        progress("downloading", 0.03)
        return {"path": path, "title": "Тестовая лекция", "duration_seconds": 0.1, "source_url": url}

    class Pipeline:
        def process(self, path, language, progress):
            assert path.read_bytes() == wav_bytes()
            assert language == "ru"
            return {"transcript": "Текст", "summary": "Конспект", "metadata": {}}

    monkeypatch.setattr("lecture_service.youtube.download_youtube_audio", download)
    with TestClient(create_app(settings)) as client:
        job = client.post("/api/lectures/youtube", json={"url": "https://youtu.be/BaW_jenozKc?list=ignored", "language": "ru"})
        assert job.status_code == 202
        job = job.json()
        assert job["source_url"] == VIDEO_URL and job["status"] == "queued"
        worker = Worker(settings, pipeline=Pipeline(), session_factory=client.app.state.sessions,
                        storage=client.app.state.storage)
        assert worker.run_once()
        completed = client.get(f"/api/lectures/{job['id']}").json()
        assert completed["status"] == "completed"
        assert completed["filename"] == "Тестовая лекция.wav"
        result = client.get(f"/api/lectures/{job['id']}/result").json()
        assert result["metadata"]["source_url"] == VIDEO_URL
        assert len(calls) == 1
        # Crash recovery after download must reuse the saved audio, without fetching YouTube again.
        with client.app.state.sessions.begin() as session:
            row = session.get(LectureJob, job["id"])
            row.status = "processing"
            row.lease_until = utcnow() - timedelta(seconds=1)
        assert worker.run_once()
        assert len(calls) == 1


def test_youtube_invalid_url_and_download_error(tmp_path, monkeypatch):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'jobs.db'}", data_dir=tmp_path, _env_file=None)

    def blocked(*args):
        raise YouTubeDownloadError("YouTube не разрешил загрузить видео.")

    monkeypatch.setattr("lecture_service.youtube.download_youtube_audio", blocked)
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/lectures/youtube", json={"url": "http://127.0.0.1/private"}).status_code == 422
        assert client.get("/api/lectures").json()["total"] == 0
        job = client.post("/api/lectures/youtube", json={"url": VIDEO_URL}).json()
        worker = Worker(settings, pipeline=object(), session_factory=client.app.state.sessions,
                        storage=client.app.state.storage)
        assert worker.run_once()
        failed = client.get(f"/api/lectures/{job['id']}").json()
        assert failed["status"] == "failed"
        assert failed["error"] == "YouTube не разрешил загрузить видео."


def test_additive_source_url_migration_preserves_old_job(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine, sessions = create_database(url)
    with sessions.begin() as session:
        session.add(LectureJob(id="old-job", filename="old.wav", object_key="old/audio.wav"))
    with engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE lecture_jobs DROP COLUMN source_url")
    engine.dispose()
    migrated, sessions = create_database(url)
    try:
        assert "source_url" in {item["name"] for item in inspect(migrated).get_columns("lecture_jobs")}
        with sessions() as session:
            original = session.scalar(select(LectureJob).where(LectureJob.id == "old-job"))
            assert original.filename == "old.wav" and original.source_url is None
    finally:
        migrated.dispose()


def test_youtube_graceful_stop_after_storage_reuses_audio_on_restart(tmp_path, monkeypatch):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'jobs.db'}", data_dir=tmp_path, _env_file=None)
    downloads = []

    def download(url, directory, settings, progress):
        downloads.append(url)
        path = directory / "BaW_jenozKc.wav"
        path.write_bytes(wav_bytes())
        return {"path": path, "title": "Лекция", "duration_seconds": 0.1}

    class Pipeline:
        calls = 0

        def process(self, path, language, progress):
            assert path.read_bytes() == wav_bytes()
            self.calls += 1
            if self.calls == 1:
                worker.stop_event.set()
                progress("transcribing", 0.1)
                raise AssertionError("Interrupted worker continued processing")
            return {"transcript": "Текст", "summary": "Конспект", "metadata": {}}

    monkeypatch.setattr("lecture_service.youtube.download_youtube_audio", download)
    with TestClient(create_app(settings)) as client:
        job = client.post("/api/lectures/youtube", json={"url": VIDEO_URL}).json()
        worker = Worker(settings, pipeline=Pipeline(), session_factory=client.app.state.sessions,
                        storage=client.app.state.storage)
        assert worker.run_once()
        with client.app.state.sessions() as session:
            interrupted = session.get(LectureJob, job["id"])
            assert interrupted.status == "queued" and interrupted.object_key.endswith("/audio.wav")
            assert interrupted.attempts == 0 and interrupted.error is None
        assert not list(tmp_path.glob("youtube-*"))
        worker.stop_event.clear()
        assert worker.run_once()
        assert client.get(f"/api/lectures/{job['id']}").json()["status"] == "completed"
        assert len(downloads) == 1
