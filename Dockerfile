FROM python:3.11-slim-bookworm 
# Using -slim reduces image size by ~400MB, making deploys faster.

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for Scipy/SKLearn
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Combined install to save memory and reduce layer count
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Render provides the $PORT env var automatically
CMD gunicorn --bind 0.0.0.0:$PORT app:app