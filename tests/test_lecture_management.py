import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from lecture_service.config import Settings
from lecture_service.db import LectureJob
from lecture_service.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'jobs.db'}", data_dir=tmp_path,
                        storage_backend="local", _env_file=None)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def seed_job(client, status="completed", pending=False):
    job_id = str(uuid4())
    key = f"{job_id}/pending" if pending else f"{job_id}/audio.wav"
    result = {"transcript": "Полная расшифровка.", "summary": "Краткий конспект.",
              "metadata": {"source_title": "Исходное название"}}
    if not pending:
        source = client.app.state.settings.data_dir / "source.wav"
        source.write_bytes(b"original audio fixture")
        client.app.state.storage.put_file(key, source)
    with client.app.state.sessions.begin() as session:
        job = LectureJob(id=job_id, filename="Исходное название.wav", object_key=key,
                         source_url="https://www.youtube.com/watch?v=BaW_jenozKc" if pending else None,
                         status=status, stage=status, progress=1 if status == "completed" else 0.3,
                         result=result if status == "completed" else None,
                         error="Ошибка обработки" if status == "failed" else None)
        session.add(job)
    return job_id, key, result


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_rename_persists_display_name_and_preserves_original_audio_and_result(client, status):
    job_id, key, original_result = seed_job(client, status)
    response = client.patch(f"/api/lectures/{job_id}", json={"filename": "  Лекция 2: Новое название  "})
    assert response.status_code == 200
    renamed = response.json()
    assert renamed["filename"] == "Лекция 2: Новое название" and renamed["status"] == status
    assert client.get(f"/api/lectures/{job_id}").json()["filename"] == renamed["filename"]
    assert client.get("/api/lectures").json()["items"][0]["filename"] == renamed["filename"]
    with client.app.state.sessions() as session:
        row = session.get(LectureJob, job_id)
        assert row.object_key == key
        assert row.result == (original_result if status == "completed" else None)
    assert client.app.state.storage._path(key).read_bytes() == b"original audio fixture"
    if status == "completed":
        assert client.get(f"/api/lectures/{job_id}/result").json() == original_result
        markdown = client.get(f"/api/lectures/{job_id}/download?format=md").text
        assert markdown.startswith("# Лекция 2: Новое название\n")
        assert original_result["summary"] in markdown and original_result["transcript"] in markdown
        assert client.get(f"/api/lectures/{job_id}/download?format=json").json() == original_result


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_delete_removes_original_object_and_database_result(client, status):
    job_id, key, _ = seed_job(client, status)
    other_id, other_key, _ = seed_job(client)
    response = client.delete(f"/api/lectures/{job_id}")
    assert response.status_code == 204 and response.content == b""
    assert not client.app.state.storage._path(key).exists()
    assert client.app.state.storage._path(other_key).is_file()
    assert client.get(f"/api/lectures/{other_id}").status_code == 200
    assert client.get(f"/api/lectures/{job_id}").status_code == 404
    assert client.get(f"/api/lectures/{job_id}/result").status_code == 404
    assert client.delete(f"/api/lectures/{job_id}").status_code == 404
    assert client.get("/api/lectures").json()["total"] == 1
    with client.app.state.sessions() as session:
        assert session.get(LectureJob, job_id) is None


def test_storage_failure_preserves_database_result_and_returns_safe_retryable_error(client, monkeypatch):
    job_id, key, result = seed_job(client)
    original_delete = client.app.state.storage.delete

    def unavailable(object_key):
        assert object_key == key
        raise RuntimeError("secret storage credentials https://private.example/?token=secret")

    monkeypatch.setattr(client.app.state.storage, "delete", unavailable)
    response = client.delete(f"/api/lectures/{job_id}")
    assert response.status_code == 503
    assert "secret" not in response.text and "private.example" not in response.text
    assert client.get(f"/api/lectures/{job_id}/result").json() == result
    assert client.app.state.storage._path(key).is_file()
    monkeypatch.setattr(client.app.state.storage, "delete", original_delete)
    assert client.delete(f"/api/lectures/{job_id}").status_code == 204


def test_failed_youtube_pending_job_deletes_without_storage_call(client, monkeypatch):
    job_id, _, _ = seed_job(client, "failed", pending=True)

    def unexpected_delete(_):
        pytest.fail("Pending YouTube jobs have no stored audio object")

    monkeypatch.setattr(client.app.state.storage, "delete", unexpected_delete)
    assert client.delete(f"/api/lectures/{job_id}").status_code == 204
    assert client.get(f"/api/lectures/{job_id}").status_code == 404


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_active_jobs_cannot_be_renamed_or_deleted(client, status, monkeypatch):
    job_id, key, _ = seed_job(client, status)
    monkeypatch.setattr(client.app.state.storage, "delete", lambda _: pytest.fail("Active audio must not be deleted"))
    assert client.patch(f"/api/lectures/{job_id}", json={"filename": "Новое"}).status_code == 409
    assert client.delete(f"/api/lectures/{job_id}").status_code == 409
    original = client.get(f"/api/lectures/{job_id}").json()
    assert original["status"] == status and original["filename"] == "Исходное название.wav"
    assert client.app.state.storage._path(key).is_file()


@pytest.mark.parametrize("filename", [
    "", "   ", ".", "..", "../lecture", "a/b", "a\\b", "C:lecture", "lecture\nname",
    "lecture\x00name", "lecture\x85name", "lecture\u2028name", "x" * 256, None, 42, {}, [],
])
def test_invalid_display_names_are_rejected_without_mutating_job(client, filename):
    job_id, _, _ = seed_job(client)
    response = client.patch(f"/api/lectures/{job_id}", json={"filename": filename})
    assert response.status_code == 422
    assert client.get(f"/api/lectures/{job_id}").json()["filename"] == "Исходное название.wav"


def test_longest_name_accepted_and_unknown_fields_or_missing_jobs_rejected(client):
    job_id, _, _ = seed_job(client)
    response = client.patch(f"/api/lectures/{job_id}", json={"filename": "  " + "я" * 255 + "  "})
    assert response.status_code == 200 and len(response.json()["filename"]) == 255
    assert client.patch(f"/api/lectures/{job_id}", json={"filename": "Новое", "status": "queued"}).status_code == 422
    assert client.patch("/api/lectures/missing", json={"filename": "Новое"}).status_code == 404
    assert client.delete("/api/lectures/missing").status_code == 404


def test_rename_content_length_limit_rejects_body_before_name_validation(client):
    job_id, _, _ = seed_job(client)
    response = client.patch(f"/api/lectures/{job_id}", json={"filename": "x" * (16 * 1024)})
    assert response.status_code == 413
    assert client.get(f"/api/lectures/{job_id}").json()["filename"] == "Исходное название.wav"
    assert client.patch(f"/api/lectures/{job_id}", json={"filename": "Обычное название"}).status_code == 200


def test_rename_chunked_body_limit_rejects_stream_without_changing_job(client):
    job_id, _, _ = seed_job(client)

    def chunks():
        yield b'{"filename":"'
        yield b"x" * (16 * 1024)
        yield b'"}'

    response = client.patch(f"/api/lectures/{job_id}", content=chunks(),
                            headers={"Content-Type": "application/json"})
    assert "content-length" not in response.request.headers
    assert response.status_code == 413
    assert client.get(f"/api/lectures/{job_id}").json()["filename"] == "Исходное название.wav"
    assert client.patch(f"/api/lectures/{job_id}", json={"filename": "Обычное название"}).status_code == 200


@pytest.mark.parametrize("path", ["/other/resource", "/api/lectures/id/result", "/api/lectures/id/download"])
def test_rename_body_limit_does_not_apply_to_unrelated_patch_resources(client, path):
    response = client.patch(path, json={"filename": "x" * (16 * 1024)})
    assert response.status_code in {404, 405}


@pytest.mark.parametrize("second_operation", ["rename", "delete"])
def test_concurrent_mutations_wait_for_delete_then_observe_missing_row(client, monkeypatch, second_operation):
    job_id, _, _ = seed_job(client)
    deleting = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    calls = []
    original_delete = client.app.state.storage.delete

    def slow_delete(key):
        calls.append(key)
        deleting.set()
        assert release.wait(timeout=5)
        original_delete(key)

    def second_request():
        second_started.set()
        if second_operation == "rename":
            return client.patch(f"/api/lectures/{job_id}", json={"filename": "Новое"})
        return client.delete(f"/api/lectures/{job_id}")

    monkeypatch.setattr(client.app.state.storage, "delete", slow_delete)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.delete, f"/api/lectures/{job_id}")
        try:
            assert deleting.wait(timeout=5)
            second = executor.submit(second_request)
            assert second_started.wait(timeout=5)
            assert not second.done()
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 204
        assert second.result(timeout=5).status_code == 404
    assert len(calls) == 1
