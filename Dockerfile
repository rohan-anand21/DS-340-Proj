# Use a slim Python base image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Install OS packages needed by TensorFlow/Transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application (models, scripts, etc.)
COPY . .

# Expose the port the app listens on inside the container
EXPOSE 7860

# Start FastAPI via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
