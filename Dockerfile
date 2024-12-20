# CRISP - production container image.
# Works on Hugging Face Spaces, Render, Fly.io, Railway, Cloud Run, or
# plain Docker.

FROM python:3.11-slim

# libgl1 + libglib2.0-0 are needed by OpenCV even in headless builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch the model weights (~655 MB) from the GitHub Release. They are not
# in the repository, so the image build pulls them in. Override the source
# with CRISP_MODELS_REPO / CRISP_MODELS_TAG at build time if you host them
# elsewhere.
ARG CRISP_MODELS_REPO=Saif-khann/CRISP
ARG CRISP_MODELS_TAG=models-v1
RUN CRISP_MODELS_REPO="$CRISP_MODELS_REPO" CRISP_MODELS_TAG="$CRISP_MODELS_TAG" \
    python download_models.py \
    && python download_models.py --check

# Run as a non-root user. Hugging Face Spaces expects uid 1000
# specifically; other platforms are happy with any non-root uid.
RUN useradd --create-home --uid 1000 crisp \
    && mkdir -p /app/static/uploads /app/data \
    && chown -R crisp:crisp /app
USER crisp

ENV PYTHONUNBUFFERED=1 \
    CRISP_ENV=production \
    CRISP_DB_PATH=/app/data/crisp.db \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/healthz" || exit 1

# One worker: each worker loads its own ~841 MB copy of the TensorFlow
# models, so additional workers multiply memory rather than throughput.
# Scale with more instances, not more workers on one instance.
# The long timeout covers first-request model loading.
CMD gunicorn --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --timeout 180 \
    --access-logfile - \
    --error-logfile - \
    app:app
