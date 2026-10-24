# Single-stage CPU build for Render free tier deploy.
# Models + Stanza pipelines baked into image (no Disk add-on on free tier).
FROM python:3.11-slim AS final

WORKDIR /app

# Native build deps for TensorFlow / numpy wheels that fall back to source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (CPU-only; strip CUDA extras to fit Render free tier) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && sed 's/\[and-cuda\]//g' requirements.txt > requirements.cpu.txt \
    && pip install --no-cache-dir -r requirements.cpu.txt

# ---- Stanza pipelines (en + zh-hans) baked at build time ----
# Avoids first-request download cost at runtime. Layer is ~700 MB but cacheable
# across builds because it only invalidates when stanza version changes.
RUN python -c "import stanza; stanza.download('en', verbose=False); stanza.download('zh-hans', verbose=False)"

# ---- Application code ----
COPY regional_cnn_lstm.py .
COPY app/ ./app/

# ---- Models + runtime cache (vocab + label_stats only; parsed_trees excluded via .dockerignore) ----
COPY models/ ./models/
COPY cache/ ./cache/

# ---- Runtime env ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=1 \
    BASE_DIR=/app \
    CUDA_VISIBLE_DEVICES=-1 \
    PORT=8000

EXPOSE 8000

# Render injects $PORT (typically 10000). Use shell form so $PORT expands.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell-form CMD so ${PORT} expands at container start.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
