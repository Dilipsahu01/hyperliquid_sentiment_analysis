FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if required by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p outputs/figures outputs/reports

# Copy the rest of the project
COPY . .

# Command to run the application
CMD ["python", "main.py"]
