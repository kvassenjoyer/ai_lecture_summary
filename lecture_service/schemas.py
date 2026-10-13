from datetime import UTC, datetime
from typing import Literal
from unicodedata import category

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    language: str | None
    source_url: str | None = None
    status: Literal["queued", "processing", "completed", "failed"]
    stage: str
    progress: float
    error: str | None
    created_at: datetime
    updated_at: datetime
    duration_seconds: float | None

    @field_serializer("created_at", "updated_at")
    def serialize_time(self, value: datetime) -> str:
        return value.replace(tzinfo=UTC).isoformat()


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int


class YouTubeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    language: str | None = Field(default=None, max_length=12)


class RenameLectureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255, strict=True)

    @field_validator("filename", mode="before")
    @classmethod
    def validate_display_name(cls, value):
        if not isinstance(value, str):
            return value
        value = value.strip()
        if (value in {"", ".", ".."} or "/" in value or "\\" in value
                or any(category(character) in {"Cc", "Cs", "Zl", "Zp"} for character in value)):
            raise ValueError("Укажите непустое название без путей и управляющих символов.")
        if len(value) >= 2 and value[0].isascii() and value[0].isalpha() and value[1] == ":":
            raise ValueError("Название не должно быть путём к файлу.")
        return value
