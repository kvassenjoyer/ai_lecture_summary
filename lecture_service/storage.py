import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .config import Settings


class LocalStorage:
    def __init__(self, settings: Settings):
        self.root = (settings.data_dir / "uploads").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        target = (self.root / key).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError("Invalid object key")
        return target

    def put_file(self, key: str, source: Path) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    @contextmanager
    def materialize(self, key: str):
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError("Uploaded audio is missing")
        yield path


class S3Storage:
    def __init__(self, settings: Settings):
        import boto3
        from botocore.config import Config
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3", endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}),
        )
        self.region = settings.s3_region

    def _ensure_bucket(self):
        from botocore.exceptions import ClientError
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            arguments = {"Bucket": self.bucket}
            if self.region != "us-east-1":
                arguments["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            try:
                self.client.create_bucket(**arguments)
            except ClientError as create_exc:
                if create_exc.response.get("Error", {}).get("Code") != "BucketAlreadyOwnedByYou":
                    raise

    def put_file(self, key: str, source: Path) -> None:
        self._ensure_bucket()
        self.client.upload_file(str(source), self.bucket, key)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    @contextmanager
    def materialize(self, key: str):
        with TemporaryDirectory(prefix="lecture-audio-") as directory:
            target = Path(directory) / ("audio" + Path(key).suffix)
            self.client.download_file(self.bucket, key, str(target))
            yield target


def create_storage(settings: Settings):
    return S3Storage(settings) if settings.storage_backend == "s3" else LocalStorage(settings)
