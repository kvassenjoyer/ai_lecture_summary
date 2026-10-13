from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    # UTC without tzinfo is portable between SQLite and PostgreSQL timestamp columns.
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class LectureJob(Base):
    __tablename__ = "lecture_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    object_key: Mapped[str] = mapped_column(String(255))
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(12), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(80), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    worker_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)


def create_database(database_url: str):
    url = make_url(database_url)
    sqlite = url.get_backend_name() == "sqlite"
    if sqlite and url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"connect_args": {"check_same_thread": False, "timeout": 30}} if sqlite else {"pool_pre_ping": True}
    if sqlite and url.database in (None, "", ":memory:"):
        from sqlalchemy.pool import StaticPool
        kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, **kwargs)
    if sqlite:
        @event.listens_for(engine, "connect")
        def sqlite_setup(connection, _):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    # API and worker may boot together; serialize initial schema creation.
    with engine.begin() as connection:
        if sqlite:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        elif engine.dialect.name == "postgresql":
            connection.execute(text("SELECT pg_advisory_xact_lock(860715299)"))
        Base.metadata.create_all(connection)
        # Additive migration for databases created before YouTube ingestion existed.
        columns = {column["name"] for column in inspect(connection).get_columns("lecture_jobs")}
        if "source_url" not in columns:
            connection.exec_driver_sql("ALTER TABLE lecture_jobs ADD COLUMN source_url TEXT")
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
