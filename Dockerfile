# Dockerfile for Hugging Face Spaces (Docker SDK).
# Builds a single container that serves both the FastAPI backend and the static
# frontend on port 7860 (the port HF expects, set via app_port in README.md).

FROM python:3.11-slim

# System libs for Pillow / OpenCV-style image decoding.
# libgl1-mesa-glx was renamed to libgl1 in Debian 13 (Trixie), which python:3.11-slim
# now uses. Keep both names? No - just use libgl1, it's the right package on modern Debian.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so layer caches well
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code + model
COPY backend ./backend
COPY frontend ./frontend
COPY skin_classifier_efficientnet_b0.pt .

# Cache directory for torchvision's ImageNet weights download.
# HF Spaces containers run as a non-root user with HOME=/home/user; make sure
# torch can write its cache there without permission errors.
ENV HOME=/home/user
RUN mkdir -p /home/user/.cache && chmod -R 777 /home/user

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
