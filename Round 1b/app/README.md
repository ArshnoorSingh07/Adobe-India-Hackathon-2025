# Adobe Hackathon 2025 – Persona-Driven Document Intelligence
# Round 1B

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

This project extracts the five most relevant and diverse sections from each PDF collection based on a given persona and job-to-be-done.  
It is **fully offline**, leaderboard-compliant, and runs efficiently on CPU after a one-time model download.

---

## Features

- Semantic section extraction using MiniLM embeddings, TF-IDF, and heading heuristics.
- No file-specific hardcoding—works on any digital PDF collection.
- Fully offline: All models and code run locally, with zero internet at runtime.
- Leaderboard-ready output (JSON format).

---

## 🚩 IMPORTANT: Download Models Before Use

Before running the extractor, **download all models locally** with:
```
python download_models.py
```
This will download the MiniLM encoder and Flan-T5-small summarizer into `/models`.

---

## Directory Structure & Usage

```
app/
├── input/
│   ├── Collection 1/PDFs/
│   ├── Collection 2/PDFs/
│   └── Collection 3/PDFs/
├── output/
├── models/
├── src/
│   └── extractor.py
├── requirements.txt
├── download_models.py
├── Dockerfile
├── approach_explanation.md
└── README.md

# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download models
python download_models.py

# Step 3: Place input PDFs (e.g., input/Collection 1/PDFs/)
# Step 4: Run extractor
python src/extractor.py

# Step 5: Check output
# Output JSONs will be generated in /output/ (one per collection)
```

---

## Docker Usage

```
# Build Docker image:
docker build --platform linux/amd64 -t persona-extractor:latest .

# Run in PowerShell:
docker run --rm -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" --network none persona-extractor:latest
```

---

## Output

For each collection, you get a JSON with:
- Metadata (input files, persona, job, timestamp)
- Top 5 extracted sections (file, heading, rank, page)
- Summarized/cleaned section text

---

## Troubleshooting

- If file not found: Check folder spelling and structure.
- If model not found: Re-run `python download_models.py`.

---

## Technical Details

See `approach_explanation.md` for pipeline and logic details.

---

*For queries, clarifications, or collaborations, please contact us at the above email.*