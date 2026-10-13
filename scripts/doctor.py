"""Print actionable local readiness checks without loading models or changing data."""
from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path

from lecture_service.config import get_settings


def main() -> None:
    settings = get_settings()
    report = {"ffmpeg": shutil.which(settings.ffmpeg_path), "storage": settings.storage_backend,
              "database": settings.database_url.split(":", 1)[0], "models": []}
    for package in ("torch", "transformers", "bitsandbytes", "fastapi"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = "MISSING: run uv sync --extra dev"
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    except ImportError:
        report["cuda_available"] = False
    from huggingface_hub import try_to_load_from_cache
    for repo in (settings.whisper_model, settings.qwen_model):
        config = try_to_load_from_cache(repo, "config.json", cache_dir=settings.hf_home / "hub")
        report["models"].append({"model": repo, "cached_config": config if isinstance(config, str) else None})
    manifest = settings.hf_home / "manifest.json"
    report["manifest"] = str(manifest.resolve()) if manifest.is_file() else "MISSING: run scripts/download_models.py"
    if manifest.is_file():
        report["downloaded_checkpoints"] = [item["repo_id"] for item in json.loads(manifest.read_text())["models"]]
    report["disk_free_gb"] = round(shutil.disk_usage(Path.cwd()).free / 1e9, 1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
