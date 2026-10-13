import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lecture_service.youtube import YouTubeDownloadError, download_youtube_audio, validate_youtube_url

VIDEO_ID = "jNQXAC9IVRw"
CANONICAL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.mark.parametrize("url", [
    CANONICAL,
    f"https://youtube.com/watch?v={VIDEO_ID}&list=PL123&index=2&t=42#hello",
    f"https://m.youtube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com:443/watch?v={VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}?si=tracking#fragment",
    f"https://www.youtu.be/{VIDEO_ID}",
    f"https://www.youtube.com/shorts/{VIDEO_ID}",
    f"https://www.youtube.com/live/{VIDEO_ID}",
])
def test_supported_video_urls_are_canonicalized_to_one_video(url):
    assert validate_youtube_url(url) == CANONICAL


@pytest.mark.parametrize("url", [
    "", "not a url", "https://youtube.com/playlist?list=PL123",
    "https://youtube.com/@example", "https://youtube.com/watch", "https://youtube.com/watch?v=short",
    f"http://youtube.com/watch?v={VIDEO_ID}", f"https://evil.com/watch?v={VIDEO_ID}",
    f"https://youtube.com.evil.com/watch?v={VIDEO_ID}",
    f"https://youtube.com@evil.com/watch?v={VIDEO_ID}",
    f"https://user:password@youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com:80/watch?v={VIDEO_ID}", f"https://youtube.com:bad/watch?v={VIDEO_ID}",
    f"https://youtube.com./watch?v={VIDEO_ID}", f"https://127.0.0.1/watch?v={VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}/another", f"https://youtu.be/{VIDEO_ID}%2f",
    f"https://youtube.com/watch?v={VIDEO_ID}&v={VIDEO_ID}",
    f"https://youtube.com\n/watch?v={VIDEO_ID}", f"https://youtube.com\\@evil.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/shorts/{VIDEO_ID}/extra", "file:///C:/secrets.txt",
])
def test_invalid_non_video_or_ambiguous_urls_are_rejected(url):
    with pytest.raises(ValueError):
        validate_youtube_url(url)


@pytest.fixture
def downloader(monkeypatch, tmp_path):
    state = SimpleNamespace(options=None, url=None, error=None, skipped=False, progress=None,
                            filename=None, content=b"audio", info={"id": VIDEO_ID, "title": "Тема\nлекции",
                            "duration": 42.5, "vcodec": "none", "availability": "public"})

    class FakeYoutubeDL:
        def __init__(self, options):
            state.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            state.url = url
            assert download
            if state.error:
                raise state.error
            reason = state.options["match_filter"](state.info, incomplete=False)
            if reason:
                state.skipped = True
                return state.info
            if state.progress:
                state.options["progress_hooks"][0](state.progress)
            filename = state.filename or state.options["outtmpl"].replace("%(ext)s", "webm")
            Path(filename).write_bytes(state.content)
            return {**state.info, "requested_downloads": [{"filepath": filename,
                                                            "vcodec": state.info.get("vcodec", "none")}]}

        def prepare_filename(self, info):
            return state.options["outtmpl"].replace("%(ext)s", "webm")

    module = ModuleType("yt_dlp")
    module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    monkeypatch.setattr("lecture_service.youtube.resolve_ffmpeg", lambda configured: "C:/tools/ffmpeg.exe")
    state.settings = SimpleNamespace(max_upload_mb=1, max_audio_seconds=120, ffmpeg_path="ffmpeg")
    state.directory = tmp_path / "download"
    return state


def test_download_is_audio_only_safe_named_and_bounded(downloader):
    state = downloader
    state.info["title"] = "../../Unsafe title\nТема лекции"
    state.progress = {"downloaded_bytes": 50, "total_bytes": 100}
    updates = []
    result = download_youtube_audio(f"https://youtu.be/{VIDEO_ID}?list=PL123", state.directory,
                                    state.settings, lambda stage, fraction: updates.append((stage, fraction)))
    assert state.url == CANONICAL
    assert state.options["format"] == "bestaudio"
    assert state.options["noplaylist"] is True
    assert state.options["maxfilesize"] == 1024 * 1024
    assert state.options["socket_timeout"] == 30
    assert "cookiefile" not in state.options and "cookiesfrombrowser" not in state.options
    assert result["path"].name == f"{VIDEO_ID}.webm"
    assert result["path"].parent == state.directory.resolve()
    assert result["duration_seconds"] == 42.5
    assert result["source_url"] == CANONICAL
    assert "\n" not in result["title"]
    assert all(stage == "downloading" and 0.01 <= fraction <= 0.04 for stage, fraction in updates)
    assert updates[-1] == ("downloading", 0.04)


@pytest.mark.parametrize("fields, message", [
    ({"duration": 121}, "лимит"), ({"is_live": True}, "трансляции"),
    ({"live_status": "is_upcoming"}, "премьеры"), ({"availability": "private"}, "общедоступная"),
    ({"_type": "playlist", "entries": []}, "Плейлисты"),
    ({"duration": float("nan")}, "длительность"), ({"duration": float("inf")}, "длительность"),
    ({"duration": -1}, "длительность"), ({"duration": 0}, "длительность"),
    ({"duration": "invalid"}, "длительность"),
])
def test_metadata_rejection_happens_before_download(downloader, fields, message):
    state = downloader
    state.info.update(fields)
    with pytest.raises(YouTubeDownloadError, match=message):
        download_youtube_audio(CANONICAL, state.directory, state.settings)
    assert state.skipped
    assert not list(state.directory.iterdir())


def test_unknown_duration_downloads_then_uses_bounded_decode(downloader, monkeypatch):
    state = downloader
    state.info["duration"] = None
    calls = []

    def normalize(path, target, *, ffmpeg_path, max_seconds):
        assert path.is_file()
        calls.append((ffmpeg_path, max_seconds))
        target.write_bytes(b"temporary normalized audio")
        return 41.5

    monkeypatch.setattr("lecture_service.youtube.normalize_audio", normalize)
    result = download_youtube_audio(CANONICAL, state.directory, state.settings)
    assert result["duration_seconds"] == 41.5
    assert calls == [("C:/tools/ffmpeg.exe", 120)]
    assert not list(state.directory.glob("*.duration-check.wav"))


def test_oversized_download_is_deleted(downloader):
    state = downloader
    state.content = b"x" * (1024 * 1024 + 1)
    with pytest.raises(YouTubeDownloadError, match="Размер"):
        download_youtube_audio(CANONICAL, state.directory, state.settings)
    assert not list(state.directory.iterdir())


def test_download_progress_aborts_when_unknown_total_exceeds_size_limit(downloader):
    state = downloader
    state.progress = {"downloaded_bytes": 1024 * 1024 + 1}
    with pytest.raises(YouTubeDownloadError, match="Размер"):
        download_youtube_audio(CANONICAL, state.directory, state.settings)
    assert not list(state.directory.iterdir())


def test_downloader_failure_does_not_leak_cdn_urls_or_cookies(downloader):
    state = downloader
    state.error = RuntimeError("403 https://cdn.example/?secret=token cookie:password")
    with pytest.raises(YouTubeDownloadError) as caught:
        download_youtube_audio(CANONICAL, state.directory, state.settings)
    assert "https://" not in str(caught.value)
    assert "secret" not in str(caught.value) and "password" not in str(caught.value)


def test_accidental_video_format_and_wrong_identity_are_rejected(downloader):
    state = downloader
    state.info["vcodec"] = "av1"
    with pytest.raises(YouTubeDownloadError, match="отдельную аудиодорожку"):
        download_youtube_audio(CANONICAL, state.directory, state.settings)
    state.info.update(vcodec="none", id="abcdefghijk")
    with pytest.raises(YouTubeDownloadError, match="[Ии]дентификатор"):
        download_youtube_audio(CANONICAL, state.directory, state.settings)


def test_output_outside_download_directory_is_rejected(downloader, tmp_path):
    state = downloader
    state.filename = str(tmp_path / f"{VIDEO_ID}.webm")
    with pytest.raises(YouTubeDownloadError, match="безопасно"):
        download_youtube_audio(CANONICAL, state.directory, state.settings)


def test_worker_cancellation_from_progress_hook_is_not_wrapped_as_download_failure(downloader):
    from lecture_service.worker import WorkerStopping

    downloader.progress = {"downloaded_bytes": 50, "total_bytes": 100}

    def progress(stage, fraction):
        if fraction > 0.01:
            raise WorkerStopping("stop requested")

    with pytest.raises(WorkerStopping, match="stop requested"):
        download_youtube_audio(CANONICAL, downloader.directory, downloader.settings, progress)
