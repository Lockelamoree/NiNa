# Base image for Python
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    python3-tk \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script and PDF folder
COPY . /app

# Expose the necessary port
EXPOSE 8000

# Set environment variables for model
ENV MODEL_NAME="hf.co/LockeLamora2077/NiNa"
ENV USE_GUI="false"

# Set the entrypoint
CMD ["python", "main.py"]
