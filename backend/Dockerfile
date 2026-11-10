
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy RAG pipeline files
COPY rag_pipeline.py /app/rag_pipeline.py
COPY hybrid_search.py /app/hybrid_search.py

# Copy serialized FAISS vector database
COPY vectorstore/ /app/vectorstore/

# Copy Flask backend
COPY backend/ /app/backend/

# Configure Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose Flask API port
EXPOSE 5000

# Start Flask using Gunicorn
CMD ["gunicorn", "--chdir", "/app/backend", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "180", "app:app"]
