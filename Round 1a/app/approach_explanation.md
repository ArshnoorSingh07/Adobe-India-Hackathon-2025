
---

# Approach Explanation – Adobe India Hackathon 2025: PDF Outline Extractor

## Problem Statement

Given any PDF—digital or scanned, in English, Hindi, or Japanese—extract the document title and all H1, H2, H3 headings (with their page numbers and levels), outputting a clean, hierarchical JSON outline. Solution must be robust, modular, run offline, and require no document-specific hardcoding.

---

## Methodology

### 1. **PDF Parsing**
- For every PDF in the input folder, the script first tries to extract text and layout structure with PyMuPDF.
- For pages without selectable text (i.e., scanned/image PDFs), it falls back to Tesseract OCR (supporting English, Hindi, and Japanese if the correct language packs are present).

### 2. **Line Grouping & Feature Extraction**
- Individual spans are grouped into lines using spatial proximity (vertical and horizontal alignment).
- Each line is annotated with font size, width ratio, alpha ratio, and position features, making the logic robust to layout variations.

### 3. **Heading Detection**
- Candidate headings are identified using a combination of:
  - **Font size:** Relative to the average document font, used to find prominent text.
  - **Numbering patterns:** e.g., "2.", "3.1", "1.1.1".
  - **NLP checks:** At least 2 words, presence of nouns/adjectives (via spaCy).
  - **Exclusions:** Skips lines that are tables, form fields, repeated digits, footers, headers, or paragraphs.

### 4. **Noise Filtering**
- The script filters out lines that are repeated across pages (like "10 10 10 10", page numbers, or headers/footers).
- Table-like lines (many numeric columns) and form labels are also ignored.

### 5. **Merging Split Headings**
- Many PDFs break long headings across lines. The script merges consecutive lines with similar font size, vertical position, and short length to form a single heading when appropriate.
- Also merges section numbers (like "2.3") with following short text lines.

### 6. **Heading Level Assignment**
- Explicitly numbered headings (like "1.", "1.1") are mapped to H1/H2/H3 based on the depth of the number.
- Remaining headings are assigned levels using font size clustering (using KMeans if there are many distinct font sizes).

### 7. **Multilingual Support**
- For image-based pages, Tesseract is invoked in multilingual mode (`eng+hin+jpn`).
- All heading logic remains language-agnostic and does not require language-specific hardcoding.

### 8. **Output**
- Each processed PDF outputs a JSON containing the title and a flat list of headings, each with `level`, `text`, and `page` keys, as required by the hackathon.

---

## Performance & Compliance

- **Runtime:** <10 seconds for a 50-page PDF on CPU
- **Model/Dependency Size:** <200MB (no external models)
- **No Internet Required:** All dependencies are in Docker
- **Robustness:** No hardcoded logic for any specific document or page

---

## Why This Approach Works

- Combining font/visual layout, linguistic heuristics, and multilingual OCR ensures the extractor is resilient to most real-world PDF quirks.
- Strict noise filtering (including repeated digits and footers/headers) improves accuracy.
- The merging of split headings boosts recall for PDFs with complex or artistic formatting.

---

## Limitations & Future Directions

- Headings in extremely decorative fonts or with very poor OCR may be missed.
- Can be extended with deep-learning based heading detection (if allowed within time/model constraints) for even better performance.