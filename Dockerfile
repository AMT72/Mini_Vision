FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /app/static/results /app/model_cache && \
    chown -R user:user /app

USER user

ENV HOME=/home/user \
    HF_HOME=/app/model_cache

EXPOSE 7860

CMD ["python", "app.py"]
