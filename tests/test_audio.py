import wave
from pathlib import Path

import numpy as np
import pytest

from lecture_service.audio import SAMPLE_RATE, AudioError, iter_audio_chunks, normalize_audio, resolve_ffmpeg


def write_wav(path: Path, seconds: float, rate: int = SAMPLE_RATE, channels: int = 1,
              silent: bool = False):
    time = np.arange(round(seconds * rate)) / rate
    samples = np.zeros_like(time) if silent else np.sin(2 * np.pi * 440 * time) * 0.2
    pcm = (samples * 32767).astype("<i2")
    if channels == 2:
        pcm = np.column_stack([pcm, pcm]).reshape(-1)
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())


def test_chunks_cover_tail_and_overlap_without_loading_whole_file(tmp_path):
    source = tmp_path / "long.wav"
    write_wav(source, 63)
    chunks = list(iter_audio_chunks(source))
    assert [(item.start, item.end) for item in chunks] == [(0, 30), (26, 56), (52, 63)]
    assert all(item.samples.dtype == np.float32 for item in chunks)
    assert all(item.samples.size <= 30 * SAMPLE_RATE for item in chunks)
    assert np.array_equal(chunks[0].samples[26 * SAMPLE_RATE:], chunks[1].samples[:4 * SAMPLE_RATE])
    assert all(not item.silent for item in chunks)


def test_digital_silence_is_detected(tmp_path):
    source = tmp_path / "silence.wav"
    write_wav(source, 1, silent=True)
    assert all(item.silent for item in iter_audio_chunks(source))


def test_normalization_stereo_resampling_and_duration(tmp_path):
    source, target = tmp_path / "stereo.wav", tmp_path / "mono.wav"
    write_wav(source, 1.5, rate=44100, channels=2)
    duration = normalize_audio(source, target)
    assert duration == pytest.approx(1.5, abs=0.01)
    with wave.open(str(target), "rb") as output:
        assert (output.getnchannels(), output.getframerate(), output.getsampwidth()) == (1, 16000, 2)


def test_corrupt_and_empty_files_are_rejected(tmp_path):
    source = tmp_path / "bad.mp3"
    source.write_bytes(b"not an audio file")
    with pytest.raises(AudioError, match="декодировать"):
        normalize_audio(source, tmp_path / "normalized.wav")
    assert not (tmp_path / "normalized.wav").exists()
    source.write_bytes(b"")
    with pytest.raises(AudioError, match="пуст"):
        normalize_audio(source, tmp_path / "normalized.wav")


def test_overlong_audio_is_rejected_before_decode(tmp_path):
    source, target = tmp_path / "long.wav", tmp_path / "normalized.wav"
    write_wav(source, 2)
    with pytest.raises(AudioError, match="лимит"):
        normalize_audio(source, target, max_seconds=1)
    assert not target.exists()


def test_invalid_overlap_and_non_normalized_input(tmp_path):
    source = tmp_path / "stereo.wav"
    write_wav(source, 1, channels=2)
    with pytest.raises(ValueError):
        list(iter_audio_chunks(source, 30, 30))
    with pytest.raises(AudioError):
        list(iter_audio_chunks(source))


def test_explicit_missing_ffmpeg_path_is_not_silently_ignored():
    with pytest.raises(AudioError, match="FFmpeg"):
        resolve_ffmpeg("Z:/no/such/ffmpeg.exe")


def test_silent_pipeline_rejects_without_loading_either_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lecture_service.asr import WhisperTranscriber
    from lecture_service.pipeline import LecturePipeline
    from lecture_service.summarizer import QwenSummarizer
    source = tmp_path / "silent.wav"
    write_wav(source, 1, silent=True)
    settings = SimpleNamespace(data_dir=tmp_path / "data", ffmpeg_path="ffmpeg",
                               max_audio_seconds=10, chunk_seconds=30, chunk_overlap_seconds=4)

    def forbidden_load(self):
        raise AssertionError("Model must not be loaded for digital silence")

    monkeypatch.setattr(WhisperTranscriber, "load", forbidden_load)
    monkeypatch.setattr(QwenSummarizer, "load", forbidden_load)
    with pytest.raises(AudioError, match="не найдена речь"):
        LecturePipeline(settings).process(source)
    assert not list(settings.data_dir.glob("lecture-*"))
