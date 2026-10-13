"""Bounded disk-backed decoding and overlapping PCM windows."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(ValueError):
    """The uploaded file is invalid, empty or exceeds configured limits."""


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    start: float
    end: float

    @property
    def silent(self) -> bool:
        # Gate digital silence only. This is deliberately not presented as VAD.
        return not self.samples.size or float(np.max(np.abs(self.samples))) < 1e-4


def resolve_ffmpeg(configured: str = "ffmpeg") -> str:
    candidate = shutil.which(configured)
    if candidate:
        return candidate
    if Path(configured).is_file():
        return str(Path(configured).resolve())
    if configured != "ffmpeg":
        raise AudioError(f"FFmpeg не найден: {configured}")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise AudioError("Установите FFmpeg или пакет imageio-ffmpeg.") from exc


def normalize_audio(source: Path, target: Path, *, ffmpeg_path: str = "ffmpeg",
                    max_seconds: float = 14_400) -> float:
    """Decode to mono 16 kHz PCM WAV; never decode more than limit + 1 seconds."""
    if max_seconds <= 0 or not math.isfinite(max_seconds):
        raise ValueError("max_seconds must be finite and positive")
    if not source.is_file() or not source.stat().st_size:
        raise AudioError("Аудиофайл пуст или отсутствует.")
    executable = resolve_ffmpeg(ffmpeg_path)
    flags = [executable, "-nostdin", "-hide_banner", "-protocol_whitelist", "file,pipe"]
    try:
        probe = subprocess.run(flags + ["-i", str(source.resolve())],
                               check=False, capture_output=True, text=True, errors="replace", timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Не удалось прочитать параметры аудио за 30 секунд.") from exc
    duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if duration:
        hours, minutes, seconds = map(float, duration.groups())
        if hours * 3600 + minutes * 60 + seconds > max_seconds + 0.01:
            raise AudioError(f"Длительность записи превышает лимит {max_seconds:g} секунд.")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = flags + ["-v", "error", "-y", "-i", str(source.resolve()), "-map", "0:a:0",
                       "-vn", "-sn", "-dn", "-t", str(max_seconds + 1), "-ac", "1",
                       "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", str(target.resolve())]
    try:
        decoded = subprocess.run(command, check=False, capture_output=True, text=True,
                                 errors="replace", timeout=max(120, min(1800, max_seconds)))
    except subprocess.TimeoutExpired as exc:
        target.unlink(missing_ok=True)
        raise AudioError("Превышено время преобразования аудио.") from exc
    if decoded.returncode:
        target.unlink(missing_ok=True)
        raise AudioError("Не удалось декодировать аудио. Проверьте формат и целостность файла.")
    try:
        with wave.open(str(target), "rb") as recording:
            duration_seconds = recording.getnframes() / recording.getframerate()
    except (wave.Error, EOFError) as exc:
        target.unlink(missing_ok=True)
        raise AudioError("Преобразование не создало корректный WAV.") from exc
    if duration_seconds <= 0 or duration_seconds > max_seconds + 1 / SAMPLE_RATE:
        target.unlink(missing_ok=True)
        raise AudioError("Аудиофайл пуст или превышает допустимую длительность.")
    return duration_seconds


def iter_audio_chunks(path: Path, chunk_seconds: float = 30,
                      overlap_seconds: float = 4) -> Iterator[AudioChunk]:
    if not 0 <= overlap_seconds < chunk_seconds <= 30:
        raise ValueError("Expected 0 <= overlap_seconds < chunk_seconds <= 30")
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
            raise AudioError("Ожидается моно PCM16 WAV с частотой 16000 Гц.")
        total = audio.getnframes()
        width = round(chunk_seconds * SAMPLE_RATE)
        step = round((chunk_seconds - overlap_seconds) * SAMPLE_RATE)
        if step < 1:
            raise ValueError("Chunk step is shorter than one sample")
        for offset in range(0, total, step):
            audio.setpos(offset)
            samples = np.frombuffer(audio.readframes(width), dtype="<i2").astype(np.float32) / 32768
            yield AudioChunk(samples, offset / SAMPLE_RATE, (offset + len(samples)) / SAMPLE_RATE)
            if offset + width >= total:
                break
