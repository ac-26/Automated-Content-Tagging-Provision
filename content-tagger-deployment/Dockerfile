FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for Hugging Face cache
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Copy application files
COPY app.py .
COPY dynamic_tagger.py .
COPY static/ ./static/

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]