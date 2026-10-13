"""Download one public YouTube video's audio; never accept arbitrary media URLs."""
from __future__ import annotations

import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .audio import AudioError, normalize_audio, resolve_ffmpeg

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


class YouTubeDownloadError(ValueError):
    """A safe error message suitable for the public job API."""


def validate_youtube_url(url: str) -> str:
    """Return a canonical HTTPS single-video URL without playlist/tracking data."""
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValueError("Укажите корректную ссылку на видео YouTube.")
    url = url.strip()
    if any(ord(character) <= 32 or ord(character) == 127 for character in url) or "\\" in url:
        raise ValueError("Ссылка YouTube содержит недопустимые символы.")
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise ValueError("Укажите корректную ссылку на видео YouTube.") from exc
    if (parsed.scheme != "https" or host not in _YOUTUBE_HOSTS
            or parsed.username is not None or parsed.password is not None or port not in (None, 443)):
        raise ValueError("Разрешены только HTTPS-ссылки на видео youtube.com или youtu.be.")
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path[1:] if parsed.path.startswith("/") else ""
    elif parsed.path == "/watch":
        values = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
        candidate = values[0] if len(values) == 1 else ""
    else:
        match = re.fullmatch(r"/(?:shorts|live)/([A-Za-z0-9_-]{11})", parsed.path)
        candidate = match.group(1) if match else ""
    if not _VIDEO_ID.fullmatch(candidate):
        raise ValueError("Нужна ссылка на одно видео YouTube с корректным идентификатором; плейлисты не поддерживаются.")
    return f"https://www.youtube.com/watch?v={candidate}"


def _duration(info: dict) -> float | None:
    value = info.get("duration")
    if value is None:
        return None
    if isinstance(value, bool):
        raise YouTubeDownloadError("YouTube вернул некорректную длительность записи.")
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise YouTubeDownloadError("YouTube вернул некорректную длительность записи.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise YouTubeDownloadError("YouTube вернул пустую запись или некорректную длительность.")
    return duration


def _check_metadata(info: dict, max_seconds: float) -> float | None:
    if info.get("_type") in {"playlist", "multi_video"} or "entries" in info:
        raise YouTubeDownloadError("Плейлисты не поддерживаются. Укажите ссылку на одно видео.")
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
        raise YouTubeDownloadError("Прямые трансляции и будущие премьеры не поддерживаются. Дождитесь публикации записи.")
    if info.get("availability") in {"private", "needs_auth", "premium_only", "subscriber_only"}:
        raise YouTubeDownloadError("Нужна общедоступная запись YouTube без входа в аккаунт.")
    duration = _duration(info)
    if duration is not None and duration > max_seconds:
        raise YouTubeDownloadError(f"Длительность видео превышает лимит {max_seconds:g} секунд.")
    return duration


class _QuietLogger:
    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def download_youtube_audio(url: str, directory: Path, settings,
                           progress: Callable[[str, float], None] | None = None) -> dict:
    """Fetch an audio-only stream into a caller-owned temporary job directory."""
    canonical = validate_youtube_url(url)
    video_id = canonical.rsplit("=", 1)[1]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    callback_errors = []

    def report(stage, fraction):
        if progress is not None:
            try:
                progress(stage, fraction)
            except Exception as exc:
                # yt-dlp may wrap exceptions from its hooks. Preserve worker
                # cancellation and lease loss instead of reporting download failure.
                callback_errors.append(exc)
                raise
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    ffmpeg = resolve_ffmpeg(settings.ffmpeg_path)
    try:
        import yt_dlp
    except ImportError as exc:
        raise YouTubeDownloadError("Загрузка YouTube недоступна: установите пакет yt-dlp[default].") from exc

    rejection = []

    def match_filter(info, *, incomplete=False):
        try:
            _check_metadata(info, settings.max_audio_seconds)
        except YouTubeDownloadError as exc:
            rejection.append(str(exc))
            return str(exc)
        return None

    def hook(status):
        downloaded = status.get("downloaded_bytes") or 0
        if downloaded > max_bytes:
            raise YouTubeDownloadError("Размер аудиодорожки превышает допустимый лимит.")
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        fraction = min(1, downloaded / total) if total > 0 else 0
        report("downloading", 0.01 + 0.03 * fraction)

    runtimes = {runtime: {"path": executable} for runtime in ("deno", "node")
                if (executable := shutil.which(runtime))}
    options = {
        "format": "bestaudio",  # No best/video fallback: only the audio track is downloaded.
        "noplaylist": True,
        "outtmpl": str(directory / f"{video_id}.%(ext)s"),
        "restrictfilenames": True,
        "maxfilesize": max_bytes,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 3,
        "ffmpeg_location": ffmpeg,
        "match_filter": match_filter,
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _QuietLogger(),
        "cachedir": False,
        "writethumbnail": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writeinfojson": False,
        "overwrites": False,
        "js_runtimes": runtimes,
    }
    report("downloading", 0.01)
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(canonical, download=True)
            if callback_errors:
                raise callback_errors[-1]
            if rejection:
                raise YouTubeDownloadError(rejection[-1])
            if not isinstance(info, dict):
                raise YouTubeDownloadError("Не удалось получить аудиозапись YouTube.")
            duration = _check_metadata(info, settings.max_audio_seconds)
            if info.get("id") != video_id:
                raise YouTubeDownloadError("Идентификатор полученной записи не совпадает со ссылкой.")
            selected = info.get("requested_downloads") or [info]
            if len(selected) != 1 or any(item.get("vcodec") not in (None, "none") for item in selected):
                raise YouTubeDownloadError("YouTube не предоставил отдельную аудиодорожку.")
            filename = selected[0].get("filepath") or info.get("filepath") or downloader.prepare_filename(info)
            path = Path(filename).resolve()
            if not path.is_relative_to(directory) or not path.name.startswith(video_id + "."):
                raise YouTubeDownloadError("Не удалось безопасно сохранить аудиозапись.")
            if not path.is_file() or path.stat().st_size <= 0:
                raise YouTubeDownloadError("YouTube не вернул аудиофайл. Возможно, превышен лимит размера.")
            if path.stat().st_size > max_bytes:
                path.unlink()
                raise YouTubeDownloadError("Размер аудиодорожки превышает допустимый лимит.")
            if duration is None:
                # Unknown source duration is allowed to download, then measured
                # through the same bounded decoder used by the worker pipeline.
                normalized = directory / f"{video_id}.duration-check.wav"
                try:
                    duration = normalize_audio(path, normalized, ffmpeg_path=ffmpeg,
                                               max_seconds=settings.max_audio_seconds)
                finally:
                    normalized.unlink(missing_ok=True)
    except YouTubeDownloadError:
        raise
    except AudioError as exc:
        raise YouTubeDownloadError(str(exc)) from exc
    except Exception as exc:
        if callback_errors:
            if callback_errors[-1] is exc:
                raise
            raise callback_errors[-1] from exc
        # Provider errors can contain signed CDN URLs or account hints. Store a
        # useful generic failure, rather than exposing arbitrary downloader logs.
        raise YouTubeDownloadError(
            "Не удалось скачать аудио YouTube. Видео может быть удалено, ограничено, требовать "
            "входа или недоступно из текущей сети. Проверьте ссылку и обновите yt-dlp.") from exc
    report("downloading", 0.04)
    title = str(info.get("title") or video_id).strip()[:500]
    title = " ".join(title.split()) or video_id
    return {"path": path, "title": title, "duration_seconds": float(duration),
            "source_url": canonical, "video_id": video_id}
