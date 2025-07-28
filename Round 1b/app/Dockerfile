FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && \
    apt-get install -y build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code, models, input, and docs
COPY models/ ./models/
COPY input/ ./input/
COPY src/ ./src/
COPY approach_explanation.md ./
COPY download_models.py ./
COPY README.md ./

# Output folder
RUN mkdir -p /app/output

# Run extractor (main script)
CMD ["python", "src/extractor.py"]