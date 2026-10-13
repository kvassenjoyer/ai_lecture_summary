from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from lecture_service.config import Settings
from lecture_service.db import LectureJob, create_database, utcnow
from lecture_service.storage import LocalStorage
from lecture_service.worker import Worker


@pytest.fixture
def context(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", data_dir=tmp_path, _env_file=None)
    engine, sessions = create_database(settings.database_url)
    storage = LocalStorage(settings)
    yield settings, sessions, storage
    engine.dispose()


def add_job(context, **fields):
    settings, sessions, storage = context
    source = settings.data_dir / "source.wav"
    source.write_bytes(b"RIFF audio fixture")
    job = LectureJob(filename="lecture.wav", object_key="fixture/audio.wav", **fields)
    storage.put_file(job.object_key, source)
    with sessions.begin() as session:
        session.add(job)
    return job


def test_atomic_claim_and_serial_processing(context):
    settings, sessions, storage = context
    first = add_job(context)
    second = add_job(context)
    workers = [Worker(settings, session_factory=sessions, storage=storage) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(executor.map(lambda worker: worker.claim_next(), workers))
    assert {job.id for job in claimed} == {first.id, second.id}
    assert all(job.attempts == 1 and job.worker_token and job.lease_until for job in claimed)
    assert workers[0].claim_next() is None


def test_expired_leases_recover_and_exhausted_jobs_fail(context):
    settings, sessions, storage = context
    retry = add_job(context, status="processing", lease_until=utcnow() - timedelta(seconds=1),
                    attempts=1, worker_token="old-owner")
    exhausted = add_job(context, status="processing", lease_until=utcnow() - timedelta(seconds=1),
                        attempts=settings.max_job_attempts, worker_token="old-owner")
    worker = Worker(settings, session_factory=sessions, storage=storage)
    claimed = worker.claim_next()
    assert claimed.id == retry.id and claimed.attempts == 2
    assert claimed.worker_token != "old-owner"
    with sessions() as session:
        assert session.get(LectureJob, exhausted.id).status == "failed"
    assert not worker._owned_update(retry, progress=0.8)


def test_worker_progress_monotonic_and_result_persisted(context):
    settings, sessions, storage = context
    job = add_job(context)

    class Pipeline:
        def process(self, audio_path, language, progress):
            progress("transcribing", 0.6)
            progress("summarizing", 0.2)
            with sessions() as session:
                assert session.get(LectureJob, job.id).progress == 0.6
            return {"transcript": "lecture", "summary": "summary", "duration_seconds": 12,
                    "segments": [], "metadata": {}}

    worker = Worker(settings, Pipeline(), sessions, storage)
    assert worker.run_once()
    assert not worker.run_once()
    with sessions() as session:
        finished = session.get(LectureJob, job.id)
        assert finished.status == "completed"
        assert finished.result["summary"] == "summary"
        assert finished.duration_seconds == 12
        assert finished.progress == 1 and finished.worker_token is None


def test_pipeline_failure_is_terminal_and_public_error_is_safe(context):
    settings, sessions, storage = context
    job = add_job(context)

    class BrokenPipeline:
        def process(self, *_):
            raise RuntimeError("secret path C:/private/credentials and token=secret")

    assert Worker(settings, BrokenPipeline(), sessions, storage).run_once()
    with sessions() as session:
        failed = session.get(LectureJob, job.id)
        assert failed.status == "failed" and failed.lease_until is None
        assert "secret" not in failed.error


def test_local_storage_blocks_traversal(context):
    _, _, storage = context
    with pytest.raises(ValueError):
        storage._path("../../escape.wav")


def test_stopped_worker_does_not_claim_queued_job(context):
    settings, sessions, storage = context
    job = add_job(context)
    worker = Worker(settings, pipeline=object(), session_factory=sessions, storage=storage)
    worker.stop_event.set()
    assert not worker.run_once()
    with sessions() as session:
        queued = session.get(LectureJob, job.id)
        assert queued.status == "queued" and queued.attempts == 0


def test_graceful_stop_during_pipeline_requeues_without_failure_or_retry_cost(context):
    settings, sessions, storage = context
    job = add_job(context)

    class Pipeline:
        def process(self, path, language, progress):
            progress("transcribing", 0.4)
            worker.stop_event.set()
            progress("loading_summarizer", 0.62)
            pytest.fail("Shutdown must prevent the next processing stage")

    worker = Worker(settings, Pipeline(), sessions, storage)
    assert worker.run_once()
    with sessions() as session:
        interrupted = session.get(LectureJob, job.id)
        assert interrupted.status == "queued" and interrupted.attempts == 0
        assert interrupted.error is None and interrupted.worker_token is None
        assert interrupted.lease_until is None and interrupted.progress == 0
