FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /workspace

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m libuser
USER libuser
WORKDIR /workspace

COPY --from=builder --chown=libuser:libuser /workspace /workspace
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
