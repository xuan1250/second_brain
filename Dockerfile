# syntax=docker/dockerfile:1.9

# ==============================================================================
# Stage 1: Build React 19 + Vite + Tailwind CSS v4 SPA
# ==============================================================================
FROM node:24-alpine@sha256:06d7adf3e0ef61a13f1d28682726366f97a15aa1888c934d3ecec86c33918b53 AS frontend-builder

WORKDIR /app/frontend

ENV NPM_CONFIG_UPDATE_NOTIFIER=false

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# ==============================================================================
# Stage 2: Resolve & install Python 3.13 dependencies via Astral uv (CPU PyTorch)
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca AS backend-builder

WORKDIR /app/backend

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=900 \
    UV_HTTP_RETRIES=10 \
    UV_CONCURRENT_DOWNLOADS=2

# Phase 1: Install locked third-party dependencies (cached across source edits)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# Phase 2: Copy backend application source and install project non-editable
COPY backend/ /app/backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ==============================================================================
# Stage 3: Lean production runtime (FastAPI + Docling + LangGraph + React SPA)
# ==============================================================================
FROM python:3.13-slim-bookworm@sha256:fa16c23a012723c24eef39cc755a27babbba16b711e5d1e6cf16225ad21637cd AS runtime

WORKDIR /app/backend

# Install shared libraries required by OpenCV / RapidOCR / Docling and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and pre-create storage & named-volume mount directories owned by appuser
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --shell /bin/bash --create-home appuser \
    && mkdir -p /app/data/db /app/data/raw /app/data/parsed /app/data/markdown /app/data/models \
    && chown -R appuser:appuser /app

# Copy pre-built virtualenv, backend source, and default config from backend-builder
COPY --from=backend-builder --chown=appuser:appuser /app/backend/.venv /app/backend/.venv
COPY --from=backend-builder --chown=appuser:appuser /app/backend/src /app/backend/src
COPY --from=backend-builder --chown=appuser:appuser /app/backend/config.yaml /app/backend/config.yaml

# Copy compiled React SPA static bundle from frontend-builder
COPY --from=frontend-builder --chown=appuser:appuser /app/frontend/dist /app/frontend/dist

ENV PATH="/app/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FRONTEND_DIST_DIR=/app/frontend/dist \
    DATA_DIR=/app/data \
    DB_DIR=/app/data/db \
    HF_HOME=/app/data/models/huggingface \
    TORCH_HOME=/app/data/models/torch \
    FASTEMBED_CACHE_PATH=/app/data/models/fastembed \
    OMP_NUM_THREADS=4

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/system/health || exit 1

USER appuser

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

