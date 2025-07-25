
# Adobe Hackathon 2025 – PDF Outline Extractor
# Round 1A

---

## Team

- **Team Name:** Wireshark
- **Members:**
    - Arshnoor Singh (Team Lead)
    - Harshpreet
    - Prigya Goyal
- **Contact:** arshnoorsingh.05@gmail.com

---

## Overview

This repository contains our submission for the Adobe India Hackathon 2025 “Connecting the Dots” Round 1A.

We present an intelligent, fully offline PDF outline extractor that accurately identifies the document title and all H1, H2, and H3 headings. The solution supports both digital and scanned/image-based PDFs and robustly handles multilingual documents (English, Hindi, Japanese).  
The output is a clean, hackathon-compliant JSON outline for each input PDF.

---

## Features

- **Accurate heading extraction**: Title, H1, H2, H3, with correct page numbers
- **Handles digital and scanned PDFs** (uses OCR as fallback)
- **Merges split/multi-line headings** (for long or artistic headings)
- **Multilingual**: Supports English, Hindi, Japanese (Tesseract language packs required)
- **Robust noise filtering**: Removes repeated digit lines, tables, footers, headers, and artifacts
- **Generalizes**: No file-specific or hardcoded rules
- **Offline & Dockerized**: Fully portable, no web/API/internet needed
- **Compliant**: Fast (<10s for 50 pages), <200MB dependencies

---

## Project Structure

```
app/
├── input/                     # Place PDF files here
├── output/                    # Output JSONs will appear here
├── src/
│   └── extractor.py           # Main extraction script
├── requirements.txt
├── Dockerfile
├── README.md
└── approach_explanation.md
```

---

## Quick Start

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd app
   ```

2. **Build the Docker image:**
   ```sh
   docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
   ```

3. **Run the extractor:**
   ```sh
   docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none pdf-outline-extractor:latest
   ```

   - All PDFs in `input/` will be processed automatically.
   - Output `.json` files will appear in `output/`.

4. **(Optional) Local usage for dev/testing:**
   ```sh
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python src/extractor.py
   ```

---

## Output Format

For each PDF (e.g., `file.pdf`), you’ll get `output/file.json`:

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Section Heading", "page": 1},
    {"level": "H2", "text": "Subsection", "page": 2},
    {"level": "H3", "text": "Sub-subsection", "page": 2}
  ]
}
```
Each heading includes its correct level and page number.

---

## Approach Summary

- **Hybrid pipeline**: PyMuPDF for digital PDFs, Tesseract OCR (multilingual) for scanned/image-based pages
- **Line grouping & feature extraction**: Spans are grouped into lines, extracting font size, width, and alpha ratio
- **Heading detection**: Uses font size, numbering, NLP heuristics (spaCy), and advanced filtering logic
- **Noise filtering**: Removes repeated digit lines, tables, footers, headers, paragraphs, and form labels
- **Merges split headings**: Combines multi-line headings into a single logical heading
- **Level assignment**: Assigns H1/H2/H3 using numbering and font size clustering
- **Multilingual**: Tesseract is run with English, Hindi, and Japanese support
- **Compliant**: Fast, Dockerized, no hardcoding, fully offline

Full details are provided in [`approach_explanation.md`](./approach_explanation.md).

---

*For queries, clarifications, please contact us at the above email.*