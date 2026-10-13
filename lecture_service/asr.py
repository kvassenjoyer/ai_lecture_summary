"""Lazy Hugging Face Whisper inference with explicit stage unloading."""
from __future__ import annotations

import gc
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from .audio import SAMPLE_RATE, AudioChunk


def merge_overlap(previous: str, current: str, max_words: int = 100) -> tuple[str, int]:
    """Remove only a matched suffix/prefix at an adjacent audio boundary.

    Never globally deduplicate: a lecturer may deliberately revisit an earlier idea.
    One-word matches are left intact to avoid deleting natural repeated conjunctions.
    """
    left, right = previous.split(), current.split()
    def normalize(value):
        return re.sub(r"[^\w]", "", value.casefold())
    left_norm, right_norm = list(map(normalize, left)), list(map(normalize, right))
    for count in range(min(len(left), len(right), max_words), 1, -1):
        if left_norm[-count:] == right_norm[:count] and any(right_norm[:count]):
            return " ".join(right[count:]), count
    # An overlap can contain a spelling error. Permit a single-character edit
    # inside a long word, never omitted words (a missing "не" changes meaning).
    for count in range(min(len(right), max_words, 32), 3, -1):
        prefix = right_norm[:count]
        suffix = left_norm[-count:]
        if len(suffix) != count or not prefix[0] or suffix[0] != prefix[0] or suffix[-1] != prefix[-1]:
            continue
        mismatches = [(a, b) for a, b in zip(suffix, prefix) if a != b]
        if len(mismatches) != 1:
            continue
        a, b = mismatches[0]
        if min(len(a), len(b)) < 7:
            continue
        operations = [operation for operation in SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                      if operation[0] != "equal"]
        if len(operations) == 1:
            _, a_start, a_end, b_start, b_end = operations[0]
            if max(a_end - a_start, b_end - b_start) <= 1:
                return " ".join(right[count:]), count
    return current.strip(), 0


def choose_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA запрошена, но недоступна. Проверьте PyTorch или задайте MODEL_DEVICE=cpu.")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("MODEL_DEVICE должен быть auto, cpu или cuda.")
    return requested


class WhisperTranscriber:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.processor = None
        self.device = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.device = choose_device(self.settings.model_device, torch)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        common = {"cache_dir": str(Path(self.settings.hf_home) / "hub"),
                  "local_files_only": self.settings.model_local_files_only}
        self.processor = AutoProcessor.from_pretrained(self.settings.whisper_model, **common)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.settings.whisper_model, dtype=self.dtype, low_cpu_mem_usage=True,
            use_safetensors=True, **common).to(self.device).eval()

    def transcribe(self, chunk: AudioChunk, language: str | None = None) -> str:
        if chunk.silent:
            return ""
        self.load()
        import torch
        features = self.processor(chunk.samples, sampling_rate=SAMPLE_RATE,
                                  return_tensors="pt", return_attention_mask=True)
        kwargs = {"task": "transcribe", "max_new_tokens": 440, "do_sample": False,
                  "return_timestamps": False, "num_beams": 1}
        if language:
            kwargs["language"] = language
        with torch.inference_mode():
            generated = self.model.generate(
                input_features=features.input_features.to(self.device, dtype=self.dtype),
                attention_mask=features.attention_mask.to(self.device), **kwargs)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        if generated.shape[-1] >= kwargs["max_new_tokens"] and int(generated[0, -1]) not in eos:
            raise RuntimeError("Whisper достигла лимита токенов фрагмента. Уменьшите CHUNK_SECONDS; "
                               "усечённая расшифровка не сохранена.")
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def unload(self):
        self.model = self.processor = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
