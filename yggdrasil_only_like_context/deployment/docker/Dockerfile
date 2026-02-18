# YggDrasil Docker Image
# Multi-stage build for GPU inference

# === Stage 1: Base with CUDA ===
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

# === Stage 2: Dependencies ===
FROM base AS deps

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -e ".[all]" || pip install --no-cache-dir -e .

# Optional: flash-attention for faster inference
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || true

# Optional: xformers
RUN pip install --no-cache-dir xformers 2>/dev/null || true

# === Stage 3: Application ===
FROM deps AS app

WORKDIR /app
COPY . .

# Install YggDrasil
RUN pip install --no-cache-dir -e .

# Expose ports
EXPOSE 8000  
# FastAPI serving
EXPOSE 7860  
# Gradio UI

# Default: run the API server
CMD ["python", "-m", "uvicorn", "yggdrasil.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
