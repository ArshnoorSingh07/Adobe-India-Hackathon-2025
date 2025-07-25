FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies for Tesseract OCR and language packs
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-jpn && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY src/ src/
RUN mkdir -p input output

CMD ["python", "src/extractor.py"]