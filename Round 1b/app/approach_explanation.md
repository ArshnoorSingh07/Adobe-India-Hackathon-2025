# Approach Explanation – Adobe India Hackathon 2025: Round 1B

---

## Problem

Given a collection of PDFs, extract and rank the five most relevant, diverse sections for a given persona and job-to-be-done. Output should be leaderboard-compliant JSON, generated fully offline, and use no file-specific hardcoding.

---

## Pipeline

1. **PDF Text Extraction**
   - Extract digital text using PyMuPDF.
   - Ignore scanned/image PDFs (as semantic ranking requires digital text).

2. **Heading Detection**
   - Heuristics: 2–12 words, title/uppercase, not a footer/page number/artifact.
   - Group subsequent lines as section content.

3. **Section Filtering**
   - Ignore sections with less than 15 words.

4. **Semantic Ranking**
   - Compute:
     - TF-IDF relevance to persona/job query.
     - MiniLM embedding similarity between query and (a) heading, (b) section body.
   - Weighted scoring: Embedding (0.5), Heading (0.3), TF-IDF (0.2).

5. **Top-K Selection & Diversity**
   - Pick top 5, max one per PDF (file diversity enforced).

6. **Summarization**
   - Each section summarized and grammar-corrected using local Flan-T5-small.
   - Input is truncated for model safety.

7. **Output**
   - Output includes input metadata, the five selected sections (file, heading, page, rank), and cleaned summary.

---

## Offline/Performance Notes

- All models (MiniLM, Flan-T5) are local, loaded with `download_models.py`.
- No internet at runtime; leaderboard-compliant.
- Optimized for CPU inference.

---

## Why This Works

- Combines semantic similarity, heading heuristics, and robust file diversity.
- General, document-agnostic, no hardcoded filenames or structures.
- Fast, reproducible, ready for leaderboard evaluation.

---

## Limitations

- Does not handle fully scanned PDFs without OCR.
- Can be extended with ML-based heading detection or OCR for generalized use.
