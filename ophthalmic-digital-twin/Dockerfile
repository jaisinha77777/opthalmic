# ---------- Stage 1: build the frontend ----------
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # -> /app/frontend/dist

# ---------- Stage 2: python runtime (API + static frontend) ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# System libs needed by torchvision / Pillow image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching. Use the CPU torch wheels
# to keep the image small (no CUDA).
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r backend/requirements.txt

# App code, trained model weights, and data artifacts.
COPY backend/ ./backend/
COPY data/ ./data/

# Built frontend from stage 1.
COPY --from=frontend /app/frontend/dist ./frontend/dist

EXPOSE 8000
WORKDIR /app/backend
CMD ["python", "run.py", "--mode", "serve"]
