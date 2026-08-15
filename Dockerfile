# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG INSTALL_EXTRAS="retrieval,preprocessing,ranking,parsing"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CARD_PIPELINE_RETRIEVAL_ENABLED=false \
    CARD_PIPELINE_OCR_ENABLED=false \
    CARD_PIPELINE_RAG_ENABLED=false \
    CARD_PIPELINE_ALLOW_MODEL_DOWNLOADS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY ocr_trainer ./ocr_trainer
COPY configs ./configs
COPY .env.example ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[${INSTALL_EXTRAS}]"

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "router.api:app", "--host", "0.0.0.0", "--port", "8000"]
