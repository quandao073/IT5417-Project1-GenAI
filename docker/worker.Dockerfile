# syntax=docker/dockerfile:1.7
# Worker runs corpus build, embedding and index build.
# CPU by default so the image builds anywhere; compose.gpu.yaml passes the CUDA
# index so the same Dockerfile produces the GPU image.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch==2.4.1 torchvision==0.19.1

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[worker]" open_clip_torch==2.24.0 transformers==4.44.2

COPY configs ./configs
ENTRYPOINT ["cxr"]
CMD ["--help"]
