FROM python:3.12-slim

# yt-dlp uses a JavaScript runtime for current YouTube challenges.
COPY --from=node:24-bookworm-slim /usr/local/bin/node /usr/local/bin/node

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models \
    DATA_DIR=/app/data \
    MODEL_DEVICE=cpu \
    MODEL_LOCAL_FILES_ONLY=true \
    QWEN_LOAD_IN_4BIT=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
# Install CPU PyTorch first: the development image does not need CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.10,<3" \
    && pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY lecture_service ./lecture_service
COPY scripts/download_models.py scripts/range_download.py ./scripts/
RUN pip install --no-deps . \
    && useradd --create-home --uid 10001 lecture \
    && mkdir -p /app/data /app/models \
    && chown -R lecture:lecture /app
USER lecture
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["python", "-m", "uvicorn", "lecture_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
