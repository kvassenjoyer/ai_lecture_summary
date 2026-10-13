import json
import logging
import re
import tempfile
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text
from starlette.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .db import LectureJob, create_database
from .schemas import JobListResponse, JobResponse, RenameLectureRequest, YouTubeRequest
from .storage import create_storage

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".webm", ".aac"}


class UploadLimitMiddleware:
    """Bound the body before multipart parsing can spool a large upload to disk."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        method = scope.get("method")
        path = scope.get("path", "")
        upload_request = method == "POST" and path == "/api/lectures"
        small_json_request = (
            (method == "POST" and path == "/api/lectures/youtube")
            or (method == "PATCH" and re.fullmatch(r"/api/lectures/[^/]+", path) is not None)
        )
        if scope["type"] != "http" or not (upload_request or small_json_request):
            return await self.app(scope, receive, send)
        max_bytes = min(self.max_bytes, 16 * 1024) if small_json_request else self.max_bytes
        headers = dict(scope.get("headers", []))
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            response = Response("Invalid Content-Length", status_code=400)
            return await response(scope, receive, send)
        if content_length > max_bytes:
            response = Response("Upload exceeds request size limit", status_code=413)
            return await response(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > max_bytes:
                    raise HTTPException(413, "Файл превышает лимит размера запроса.")
            return message

        return await self.app(scope, limited_receive, send)


def safe_filename(name: str | None) -> str:
    name = (name or "audio.wav").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip(" .")
    if not name:
        raise HTTPException(400, "Укажите имя аудиофайла.")
    # Keep the extension when trimming a very long name.
    suffix = Path(name).suffix.lower()
    return name[: max(1, 240 - len(suffix))] + suffix if len(name) > 240 else name


def valid_audio_header(header: bytes, suffix: str) -> bool:
    if suffix == ".wav":
        return len(header) >= 12 and header[:4] in {b"RIFF", b"RF64"} and header[8:12] == b"WAVE"
    if suffix == ".flac":
        return header.startswith(b"fLaC")
    if suffix in {".ogg", ".opus"}:
        return header.startswith(b"OggS")
    if suffix in {".m4a", ".mp4"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if suffix == ".webm":
        return header.startswith(b"\x1a\x45\xdf\xa3")
    if suffix in {".mp3", ".aac"}:
        return header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
        )
    return False


def render_markdown(job: LectureJob, result: dict) -> str:
    return (
        f"# {job.filename}\n\n"
        f"## Конспект\n\n{result.get('summary', '')}\n\n"
        f"## Расшифровка\n\n{result.get('transcript', '')}\n"
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        engine, sessions = create_database(settings.database_url)
        app.state.settings = settings
        app.state.engine = engine
        app.state.sessions = sessions
        app.state.storage = create_storage(settings)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="Лекториум", version="0.1.0", lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware, max_bytes=int(settings.max_upload_mb * 1024 * 1024) + 128 * 1024)

    def get_job(job_id: str) -> LectureJob:
        with app.state.sessions() as session:
            job = session.get(LectureJob, job_id)
            if job is None:
                raise HTTPException(404, "Лекция не найдена.")
            return job

    @app.get("/health")
    def health():
        try:
            with app.state.sessions() as session:
                session.execute(text("SELECT 1"))
        except Exception:
            logger.exception("Database health check failed")
            raise HTTPException(503, "База данных недоступна.") from None
        return {"status": "ok", "storage_backend": settings.storage_backend,
                "database_backend": app.state.engine.dialect.name}

    @app.post("/api/lectures", response_model=JobResponse, status_code=202)
    async def upload_lecture(file: Annotated[UploadFile, File()], language: Annotated[str | None, Form()] = None):
        temporary_path = None
        object_key = None
        stored = False
        try:
            filename = safe_filename(file.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(415, "Поддерживаются WAV, MP3, FLAC, OGG, OPUS, M4A, MP4, WEBM и AAC.")
            language = (language or "").strip().lower()
            if language in {"", "auto"}:
                language = None
            elif not re.fullmatch(r"[a-z]{2,3}", language):
                raise HTTPException(422, "Язык: код ISO, например ru или en, либо auto.")
            job_id = str(uuid4())
            object_key = f"{job_id}/audio{suffix}"
            upload_dir = settings.data_dir / "tmp"
            upload_dir.mkdir(parents=True, exist_ok=True)
            size = 0
            header = b""
            with tempfile.NamedTemporaryFile(dir=upload_dir, suffix=suffix, delete=False) as target:
                temporary_path = Path(target.name)
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > int(settings.max_upload_mb * 1024 * 1024):
                        raise HTTPException(413, f"Файл превышает лимит {settings.max_upload_mb:g} МБ.")
                    if len(header) < 4096:
                        header += chunk[:4096 - len(header)]
                    target.write(chunk)
            if size == 0:
                raise HTTPException(400, "Аудиофайл пуст.")
            if not valid_audio_header(header, suffix):
                raise HTTPException(415, "Содержимое файла не соответствует аудиоформату.")
            await run_in_threadpool(app.state.storage.put_file, object_key, temporary_path)
            stored = True
            def persist():
                with app.state.sessions() as session:
                    job = LectureJob(id=job_id, filename=filename, object_key=object_key, language=language)
                    session.add(job)
                    session.commit()
                    return JobResponse.model_validate(job)
            return await run_in_threadpool(persist)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Upload failed")
            if stored and object_key:
                try:
                    await run_in_threadpool(app.state.storage.delete, object_key)
                except Exception:
                    logger.exception("Could not remove orphan upload")
            raise HTTPException(503, "Не удалось сохранить аудиофайл. Повторите загрузку.") from None
        finally:
            await file.close()
            if temporary_path:
                temporary_path.unlink(missing_ok=True)

    @app.get("/api/lectures", response_model=JobListResponse)
    def list_lectures(limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0)):
        with app.state.sessions() as session:
            total = session.scalar(select(func.count()).select_from(LectureJob))
            jobs = session.scalars(select(LectureJob).order_by(LectureJob.created_at.desc())
                                   .offset(offset).limit(limit)).all()
            return JobListResponse(items=[JobResponse.model_validate(job) for job in jobs], total=total)

    @app.post("/api/lectures/youtube", response_model=JobResponse, status_code=202)
    def import_youtube(request: YouTubeRequest):
        from .youtube import validate_youtube_url
        try:
            source_url = validate_youtube_url(request.url)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        language = (request.language or "").strip().lower()
        if language in {"", "auto"}:
            language = None
        elif not re.fullmatch(r"[a-z]{2,3}", language):
            raise HTTPException(422, "Язык: код ISO, например ru или en, либо auto.")
        job_id = str(uuid4())
        with app.state.sessions.begin() as session:
            job = LectureJob(id=job_id, filename="YouTube · " + source_url.rsplit("=", 1)[-1],
                             object_key=f"{job_id}/pending", source_url=source_url, language=language)
            session.add(job)
            session.flush()
            response = JobResponse.model_validate(job)
        return response

    @app.get("/api/lectures/{job_id}", response_model=JobResponse)
    def read_lecture(job_id: str):
        return get_job(job_id)

    @contextmanager
    def locked_terminal_job(job_id: str):
        with app.state.sessions.begin() as session:
            # PostgreSQL locks just this row. SQLite has no row locks, so acquire
            # its write transaction before reading to serialize rename/delete.
            if app.state.engine.dialect.name == "sqlite":
                session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            job = session.scalar(select(LectureJob).where(LectureJob.id == job_id).with_for_update())
            if job is None:
                raise HTTPException(404, "Лекция не найдена.")
            if job.status not in {"completed", "failed"}:
                raise HTTPException(409, "Дождитесь завершения обработки лекции.")
            yield session, job

    @app.patch("/api/lectures/{job_id}", response_model=JobResponse)
    def rename_lecture(job_id: str, request: RenameLectureRequest):
        try:
            with locked_terminal_job(job_id) as (session, job):
                job.filename = request.filename
                session.flush()
                response = JobResponse.model_validate(job)
            return response
        except HTTPException:
            raise
        except Exception:
            logger.exception("Could not rename lecture %s", job_id)
            raise HTTPException(503, "Не удалось переименовать лекцию. Повторите попытку.") from None

    @app.delete("/api/lectures/{job_id}", status_code=204)
    def delete_lecture(job_id: str):
        try:
            with locked_terminal_job(job_id) as (session, job):
                # Keep the row and result if the storage operation fails. A
                # retry is safe: local unlink and S3 DeleteObject are idempotent.
                if not job.object_key.endswith("/pending"):
                    app.state.storage.delete(job.object_key)
                session.delete(job)
            return Response(status_code=204)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Could not delete lecture %s", job_id)
            raise HTTPException(503, "Не удалось удалить лекцию. Повторите попытку.") from None

    def completed_job(job_id: str):
        job = get_job(job_id)
        if job.status != "completed" or job.result is None:
            raise HTTPException(409, "Результат ещё не готов." if job.status != "failed" else "Обработка завершилась ошибкой.")
        return job

    @app.get("/api/lectures/{job_id}/result")
    def read_result(job_id: str):
        return completed_job(job_id).result

    @app.get("/api/lectures/{job_id}/download")
    def download_result(job_id: str, format: str = Query("md", pattern="^(md|txt|json)$")):
        job = completed_job(job_id)
        result = job.result
        if format == "json":
            content = json.dumps(result, ensure_ascii=False, indent=2)
            media_type = "application/json"
        elif format == "txt":
            content = f"КОНСПЕКТ\n\n{result.get('summary', '')}\n\nРАСШИФРОВКА\n\n{result.get('transcript', '')}\n"
            media_type = "text/plain"
        else:
            content = render_markdown(job, result)
            media_type = "text/markdown"
        return Response(content, media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="lecture-{job.id}.{format}"'})

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        index_file = static_dir / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return {"service": "Лекториум", "docs": "/docs"}

    return app


app = create_app()
