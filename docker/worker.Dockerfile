# syntax=docker/dockerfile:1.7
# Worker chạy ingestion/embedding. Cần GPU -> dùng kèm compose.gpu.yaml.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch cài riêng để pin đúng CUDA build; đổi cu124 -> cpu nếu chạy không GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.4.1 torchvision==0.19.1

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[worker]" open_clip_torch==2.24.0 transformers==4.44.2

COPY configs ./configs
ENTRYPOINT ["cxr"]
CMD ["--help"]
