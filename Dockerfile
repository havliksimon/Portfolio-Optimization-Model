# Portfolio Optimizer - Production Dockerfile
# Optimized for cloud deployment (Koyeb, Zeabur, etc.)
# Multi-stage build with aggressive size optimization

# =============================================================================
# Stage 1: Builder - Compile dependencies
# =============================================================================
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install production requirements (uses requirements_prod.txt for smaller size)
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Strip debug symbols from compiled extensions
RUN find /opt/venv -name "*.so" -exec strip --strip-debug {} + 2>/dev/null || true

# Clean up unnecessary files (keep sklearn for PCA/clustering)
RUN rm -rf /opt/venv/lib/python*/site-packages/*/tests \
    /opt/venv/lib/python*/site-packages/pandas/tests \
    /opt/venv/lib/python*/site-packages/numpy/tests \
    /opt/venv/lib/python*/site-packages/numpy/core/tests \
    /opt/venv/lib/python*/site-packages/numpy/random/tests \
    /opt/venv/lib/python*/site-packages/scipy/stats/tests \
    /opt/venv/lib/python*/site-packages/scipy/sparse/tests \
    /opt/venv/lib/python*/site-packages/scipy/linalg/tests \
    /opt/venv/lib/python*/site-packages/scipy/optimize/tests \
    /opt/venv/lib/python*/site-packages/matplotlib/tests \
    /opt/venv/lib/python*/site-packages/sklearn/datasets \
    /opt/venv/lib/python*/site-packages/statsmodels/tests \
    /opt/venv/share/jupyter /opt/venv/share/doc /opt/venv/share/man \
    /opt/venv/include \
    /opt/venv/lib/python*/config-* \
    /opt/venv/bin/python*-config \
    /opt/venv/lib/python*/site-packages/pip \
    /opt/venv/lib/python*/site-packages/wheel \
    /opt/venv/lib/python*/site-packages/setuptools

# =============================================================================
# Stage 2: Runtime - Minimal image
# =============================================================================
FROM python:3.11-slim as runtime

# Install runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    libpq5 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads instance

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 app:app
