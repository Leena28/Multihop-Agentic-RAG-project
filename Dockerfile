# Base image
FROM python:3.12-slim

# Setting working directory inside container
WORKDIR /app

# Install system dependencies needed by PyMuPDF
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Create directory for Qdrant data
RUN mkdir -p qdrant_storage_v2

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]