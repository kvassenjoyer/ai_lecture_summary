import logging
import math
import signal
import tempfile
import threading
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update

from .config import Settings, get_settings
from .db import LectureJob, create_database, utcnow
from .storage import create_storage

logger = logging.getLogger(__name__)


class WorkerStopping(RuntimeError):
    """Cooperative shutdown at an audio or text processing boundary."""


class Worker:
    """One serial model pipeline; jobs and ownership leases live in the database."""

    def __init__(self, settings: Settings, pipeline=None, session_factory=None, storage=None):
        self.settings = settings
        self.engine = None
        if session_factory is None:
            self.engine, session_factory = create_database(settings.database_url)
        self.sessions = session_factory
        self.storage = storage or create_storage(settings)
        self.pipeline = pipeline
        self.stop_event = threading.Event()

    def claim_next(self):
        now = utcnow()
        with self.sessions.begin() as session:
            # Recover interrupted jobs. Exhausted retries become terminal failures.
            expired = (LectureJob.status == "processing", LectureJob.lease_until < now)
            session.execute(update(LectureJob).where(*expired, LectureJob.attempts >= self.settings.max_job_attempts)
                            .values(status="failed", stage="failed", error="Обработка прервана: исчерпаны попытки восстановления.",
                                    lease_until=None, worker_token=None, updated_at=now))
            session.execute(update(LectureJob).where(*expired, LectureJob.attempts < self.settings.max_job_attempts)
                            .values(status="queued", stage="queued", progress=0, worker_token=None,
                                    lease_until=None, updated_at=now))
        # Conditional UPDATE prevents two processes from claiming the same job.
        for _ in range(10):
            token = str(uuid4())
            with self.sessions.begin() as session:
                job_id = session.scalar(select(LectureJob.id).where(LectureJob.status == "queued")
                                        .order_by(LectureJob.created_at).limit(1))
                if job_id is None:
                    return None
                claimed = session.execute(update(LectureJob).where(LectureJob.id == job_id, LectureJob.status == "queued")
                                          .values(status="processing", stage="preparing", progress=0.01,
                                                  error=None, worker_token=token, attempts=LectureJob.attempts + 1,
                                                  lease_until=now + timedelta(seconds=self.settings.lease_seconds), updated_at=now))
                if claimed.rowcount == 1:
                    return session.get(LectureJob, job_id)
        return None

    def _owned_update(self, job, **values):
        with self.sessions.begin() as session:
            changed = session.execute(update(LectureJob).where(
                LectureJob.id == job.id, LectureJob.worker_token == job.worker_token,
                LectureJob.status == "processing",
            ).values(updated_at=utcnow(), **values))
            return changed.rowcount == 1

    def run_once(self) -> bool:
        if self.stop_event.is_set():
            return False
        job = self.claim_next()
        if job is None:
            return False
        heartbeat_stop = threading.Event()
        lease_lost = threading.Event()

        def heartbeat():
            while not heartbeat_stop.wait(self.settings.heartbeat_seconds):
                try:
                    if not self._owned_update(job, lease_until=utcnow() + timedelta(seconds=self.settings.lease_seconds)):
                        lease_lost.set()
                        return
                except Exception:
                    logger.exception("Could not renew lease for %s", job.id)

        thread = threading.Thread(target=heartbeat, name="job-heartbeat", daemon=True)
        thread.start()
        last_progress = 0.01

        def check_stop():
            if self.stop_event.is_set():
                raise WorkerStopping("Worker shutdown requested")

        def progress(stage: str, fraction: float):
            nonlocal last_progress
            check_stop()
            if lease_lost.is_set():
                raise RuntimeError("Job ownership was lost")
            if math.isfinite(float(fraction)):
                last_progress = max(last_progress, min(0.99, float(fraction)))
            if not self._owned_update(job, stage=str(stage)[:80], progress=last_progress,
                                      lease_until=utcnow() + timedelta(seconds=self.settings.lease_seconds)):
                lease_lost.set()
                raise RuntimeError("Job ownership was lost")

        try:
            check_stop()
            if job.source_url and job.object_key.endswith("/pending"):
                from .youtube import download_youtube_audio
                self.settings.data_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="youtube-", dir=self.settings.data_dir) as temporary:
                    downloaded = download_youtube_audio(job.source_url, Path(temporary), self.settings, progress)
                    check_stop()
                    suffix = downloaded["path"].suffix.lower()
                    object_key = f"{job.id}/audio{suffix}"
                    self.storage.put_file(object_key, downloaded["path"])
                    title = downloaded["title"].replace("/", "-").replace("\\", "-")[:180]
                    if not self._owned_update(job, object_key=object_key, filename=title + suffix):
                        raise RuntimeError("Job ownership was lost")
                    job.object_key, job.filename = object_key, title + suffix
            check_stop()
            if self.pipeline is None:
                from .pipeline import LecturePipeline
                self.pipeline = LecturePipeline(self.settings)
            with self.storage.materialize(job.object_key) as audio_path:
                check_stop()
                result = self.pipeline.process(audio_path, job.language, progress)
            check_stop()
            if not isinstance(result, dict) or not all(isinstance(result.get(key), str) for key in ("transcript", "summary")):
                raise ValueError("Pipeline returned an invalid result")
            if job.source_url:
                result.setdefault("metadata", {}).update(source_url=job.source_url, source_title=job.filename)
            self._owned_update(job, status="completed", stage="completed", progress=1,
                               result=result, duration_seconds=result.get("duration_seconds"),
                               error=None, lease_until=None, worker_token=None)
            logger.info("Completed lecture %s", job.id)
        except WorkerStopping:
            # A normal operator shutdown is not a processing failure and must
            # not consume the crash-recovery attempt budget.
            self._owned_update(job, status="queued", stage="queued", progress=0,
                               attempts=LectureJob.attempts - 1, error=None,
                               lease_until=None, worker_token=None)
            logger.info("Returned lecture %s to the queue during shutdown", job.id)
        except Exception as exc:
            logger.exception("Failed to process lecture %s", job.id)
            from .youtube import YouTubeDownloadError
            public_error = str(exc) if isinstance(exc, YouTubeDownloadError) else (
                "Не удалось обработать запись. Проверьте, что файл содержит речь, "
                "модели скачаны и для них достаточно памяти. Подробности - в журнале worker."
            )
            self._owned_update(job, status="failed", stage="failed", error=public_error,
                               lease_until=None, worker_token=None)
        finally:
            heartbeat_stop.set()
            thread.join(timeout=5)
        return True

    def run_forever(self):
        logger.info("Worker started; processing one lecture at a time")
        try:
            while not self.stop_event.is_set():
                try:
                    worked = self.run_once()
                except Exception:
                    logger.exception("Worker queue error")
                    worked = False
                if not worked:
                    self.stop_event.wait(self.settings.poll_interval_seconds)
        finally:
            if self.engine is not None:
                self.engine.dispose()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = Worker(get_settings())
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: worker.stop_event.set())
    worker.run_forever()


if __name__ == "__main__":
    main()
