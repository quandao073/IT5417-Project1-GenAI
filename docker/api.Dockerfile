# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

FROM base AS builder
WORKDIR /build
# CPU torch first, same pattern as worker.Dockerfile: the default PyPI wheel on
# Linux is the CUDA build. The API only encodes short text queries.
RUN pip install --prefix=/install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchvision==0.19.1
COPY pyproject.toml ./
COPY src ./src
# pip cannot see the previous RUN's --prefix install, so without the extra index the
# torch==2.4.1 pin in the api extra would resolve to the CUDA wheel on PyPI; with it,
# 2.4.1+cpu wins by PEP 440 ordering.
RUN pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu \
    ".[api]" open_clip_torch==2.24.0 transformers==4.44.2

FROM base AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app configs ./configs
# HF_HOME=/app/.hf_cache (.env.example) must be writable by uid 10001, or snapshot_download
# raises PermissionError and the API stays 503. compose mounts the hf_cache volume here.
RUN mkdir -p /app/.hf_cache && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
