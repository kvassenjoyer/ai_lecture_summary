from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./data/lectures.db"
    storage_backend: Literal["local", "s3"] = "local"
    data_dir: Path = Path("data")
    s3_endpoint_url: str | None = None
    s3_bucket: str = "lectures"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "us-east-1"
    max_upload_mb: float = Field(default=500, gt=0)
    max_audio_seconds: float = Field(default=14400, gt=0)
    poll_interval_seconds: float = Field(default=1, gt=0)
    model_device: str = "auto"
    whisper_model: str = "openai/whisper-large-v3-turbo"
    qwen_model: str = "Qwen/Qwen3.5-4B"
    hf_home: Path = Path("models")
    ffmpeg_path: str = "ffmpeg"
    chunk_seconds: float = Field(default=30, gt=0, le=30)
    chunk_overlap_seconds: float = Field(default=4, ge=0)
    summary_chunk_tokens: int = Field(default=1800, ge=128)
    summary_max_new_tokens: int = Field(default=900, ge=64)
    qwen_load_in_4bit: bool = True
    model_local_files_only: bool = True
    lease_seconds: int = Field(default=180, ge=5)
    heartbeat_seconds: float = Field(default=15, gt=0)
    max_job_attempts: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_intervals(self):
        if self.chunk_overlap_seconds >= self.chunk_seconds:
            raise ValueError("CHUNK_OVERLAP_SECONDS must be smaller than CHUNK_SECONDS")
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("HEARTBEAT_SECONDS must be smaller than LEASE_SECONDS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
