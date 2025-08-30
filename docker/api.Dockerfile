# ===== API (Flask + Gunicorn) =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS api

# System deps (build tools are rarely needed with manylinux wheels, keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ca-certificates curl tini && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install CUDA-enabled torch/vision from the official index
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

# Copy dependency manifest first for better caching
# If you use pyproject.toml/poetry, swap this section accordingly.
COPY requirements.txt /app/requirements.txt


# --- Optional: install CPU-only PyTorch at build time ---
# ARG INSTALL_TORCH=0
# RUN if [ "$INSTALL_TORCH" = "1" ]; then \
#       pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
#     fi

# Base deps (add your libs in requirements.txt)
RUN sed -i '/^torch/d' /app/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY src/ /app/src/
COPY model_registry/ /app/model_registry/   
# optional; safe if absent
ENV PYTHONPATH=/app/src \
    FLASK_ENV=production \
    MODEL_REGISTRY=/app/model_registry/latest

# Non-root user
RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]

# If your Flask app uses an app factory create_app(), use --factory
# Adjust the dotted path if needed: mancala_ai.api.app:create_app()
CMD ["python","-m","gunicorn",\
     "--workers=2","--threads=8","--timeout=90",\
     "--bind=0.0.0.0:8000",\
     "--access-logfile=-","--error-logfile=-",\
     "--worker-class=gthread",\
     "wsgi:app"]


