"""One job at a time, with ASR and summarizer never resident together."""
from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .asr import WhisperTranscriber, merge_overlap
from .audio import AudioError, iter_audio_chunks, normalize_audio
from .summarizer import QwenSummarizer


class LecturePipeline:
    def __init__(self, settings):
        self.settings = settings

    def process(self, audio_path: Path, language: str | None = None,
                progress: Callable[[str, float], None] | None = None) -> dict:
        report = progress or (lambda stage, value: None)
        started = time.monotonic()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lecture-", dir=self.settings.data_dir) as work:
            report("normalizing", 0.05)
            normalized = Path(work) / "normalized.wav"
            duration = normalize_audio(audio_path, normalized,
                                       ffmpeg_path=self.settings.ffmpeg_path,
                                       max_seconds=self.settings.max_audio_seconds)
            asr = WhisperTranscriber(self.settings)
            raw_chunks, merged_chunks, segments = [], [], []
            previous_end = 0.0
            total_chunks = skipped_silence = 0
            try:
                for chunk in iter_audio_chunks(normalized, self.settings.chunk_seconds,
                                               self.settings.chunk_overlap_seconds):
                    total_chunks += 1
                    report("transcribing", 0.1 + 0.5 * chunk.start / duration)
                    if chunk.silent:
                        skipped_silence += 1
                        continue
                    text = asr.transcribe(chunk, language)
                    if not text:
                        continue
                    raw_chunks.append(text)
                    trimmed, removed = (merge_overlap(" ".join(merged_chunks[-2:]), text,
                                                       max_words=max(2, round(self.settings.chunk_overlap_seconds * 8)))
                                        if segments and chunk.start < previous_end else (text, 0))
                    if trimmed:
                        merged_chunks.append(trimmed)
                        start = chunk.start + (chunk.end - chunk.start) * removed / max(1, len(text.split()))
                        segments.append({"start": round(start, 3), "end": round(chunk.end, 3), "text": trimmed})
                    previous_end = chunk.end
            finally:
                # Releases Whisper before Qwen allocation even if inference fails.
                asr.unload()
            if not merged_chunks:
                raise AudioError("В записи не найдена речь: аудио содержит тишину или не удалось распознать текст.")
            asr_seconds = time.monotonic() - started
            merged_transcript = " ".join(merged_chunks)
            report("loading_summarizer", 0.62)
            summarizer = QwenSummarizer(self.settings)
            try:
                result = summarizer.process(merged_transcript, report)
            finally:
                summarizer.unload()
            report("saving", 0.98)
            return {"transcript": result["transcript"], "raw_transcript": "\n\n".join(raw_chunks),
                    "summary": result["summary"], "segments": segments, "duration_seconds": round(duration, 3),
                    "metadata": {**result["metadata"], "whisper_model": self.settings.whisper_model,
                                 "qwen_model": self.settings.qwen_model, "requested_language": language,
                                 "audio_chunks": total_chunks, "silent_chunks_skipped": skipped_silence,
                                 "timestamp_precision": "audio chunk bounds; overlap trimming interpolated, not word-aligned",
                                 "raw_transcript_contains_overlap": True,
                                 "merged_asr_text": merged_transcript,
                                 "asr_seconds": round(asr_seconds, 2),
                                 "processing_seconds": round(time.monotonic() - started, 2)}}
