
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

# Install CPU-only PyTorch to avoid unnecessary NVIDIA GPU libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

COPY rag_pipeline.py /app/rag_pipeline.py
COPY hybrid_search.py /app/hybrid_search.py
COPY vectorstore/ /app/vectorstore/
COPY backend/ /app/backend/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=""

EXPOSE 5000

CMD ["gunicorn", "--chdir", "/app/backend", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "180", "app:app"]
