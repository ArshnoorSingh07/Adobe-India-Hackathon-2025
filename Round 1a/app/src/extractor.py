import os
import json
import fitz
import re
import numpy as np
import spacy
from collections import Counter
from sklearn.cluster import KMeans
import pytesseract
from PIL import Image

# === CONFIG ===
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
OCR_LANGS = 'eng+hin+jpn'

nlp = spacy.load("en_core_web_sm")

def is_table_like_line(text):
    # Checks if the line looks like a table row (lots of numbers/cells)
    cells = re.split(r'\s{2,}|\t|\|', text)
    if len(cells) < 3:
        return False
    numeric_ratio = sum(bool(re.fullmatch(r'[\d.,%$\u20B9\-]+', c.strip())) for c in cells) / len(cells)
    return numeric_ratio > 0.5

def is_form_label_line(text, wf, x0):
    # Checks if this line looks like a form field label or question
    is_numbered = bool(re.match(r'^(\(?\d+[\.\)]|[a-zA-Z][\.\)])', text.strip()))
    return (wf < 0.5 and x0 < 100) or is_numbered

def is_paragraph_like(text, wf, alpha):
    # Heuristic: Looks like a long paragraph, not a heading
    word_count = len(text.strip().split())
    return word_count > 15 and wf > 0.7 and alpha > 0.6

def is_valid_title(text):
    # Returns True if the line could be a good document title
    if not text or len(text.split()) < 3:
        return False
    if len(set(text.lower().split())) < len(text.split()) // 2:
        return False
    if any(char in text for char in "¶•×©®™") or text.count(" ") == 0:
        return False
    doc = nlp(text)
    return any(tok.pos_ in ("NOUN", "PROPN") for tok in doc)

def is_footer_like(text):
    # Looks for footers, page numbers, copyright, etc.
    if not text.strip():
        return True
    patterns = [
        r'Page \d+', r'Version \d{4}', r'copyright', r'www\.', r'[0-9]{4}', r'\.{2,}', r'trademark',
        r'registered', r'\b\d+ of \d+\b'
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    if len(text) < 3 or text.count(' ') > 10:
        return True
    return False

def is_numbered_heading(text):
    # True if line starts like "1.", "2.3", etc.
    return bool(re.match(r'^\d+(\.\d+)*[\.\s]?(?=\s|$)', text.strip()))

def get_heading_level_from_number(text):
    # Returns heading level based on "1." = H1, "1.1" = H2, etc.
    m = re.match(r'^(\d+(\.\d+)*)(\s+|$)', text)
    if not m:
        return None
    depth = m.group(1).count('.')
    return f"H{min(3, depth + 1)}"

def is_valid_heading(text, ocr=False, size=None, avg_size=None, wf=None, alpha=None):
    # Returns True if the line passes all filters to be considered a heading
    if is_numbered_heading(text):
        return True
    if re.fullmatch(r'(\d+[\s]*){2,}', text.strip()): 
        return False
    if not text or len(text) < 2:
        return False
    if any(char in text for char in "¶•×©®™") or len(text.strip()) < 2:
        return False
    if len(text) > 90 or len(text) < 2:
        return False
    if ocr:
        # OCR headings: filter by font size, width, alpha ratio
        if size and avg_size and size < avg_size * 1.15:
            return False
        if wf and wf > 0.93:
            return False
        if alpha and is_paragraph_like(text, wf, alpha):
            return False
        return True
    else:
        # Digital headings: require at least 2 words, some nouns, etc.
        if len(text.split()) < 2:
            return False
        if any(char in text for char in "¶•×©®™") or text.count(" ") == 0:
            return False
        words = text.lower().split()
        if len(set(words)) < len(words) // 2:
            return False
        if len(text) > 70 or len(text) < 4:
            return False
        doc = nlp(text)
        return any(tok.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") for tok in doc) and doc[0].is_title

def filter_repeated_lines(lines, min_repeats=3, min_length=3):
    # Removes lines that are repeated many times (like headers/footers)
    text_counter = Counter(l["text"] for l in lines if len(l["text"]) >= min_length)
    total_pages = max(l["page"] for l in lines) if lines else 1
    repeated_texts = set(text for text, count in text_counter.items() if count >= min_repeats or count > 0.3 * total_pages)
    filtered = [l for l in lines if l["text"] not in repeated_texts]
    return filtered, repeated_texts

def extract_spans(pdf_path):
    # Extracts individual text spans (with font, position info) from each page
    doc = fitz.open(pdf_path)
    spans = []
    for page in doc:
        page_width = page.rect.width
        text_blocks = page.get_text("dict")["blocks"]
        has_text = any(block.get("type", 0) == 0 and block.get("lines") for block in text_blocks)

        if has_text:
            # Digital PDF page
            for block in text_blocks:
                if block.get("type", 0) != 0:
                    continue
                full_text = " ".join(span["text"].strip() for line in block.get("lines", []) for span in line.get("spans", []))
                if is_table_like_line(full_text):
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        spans.append({
                            "text": text,
                            "size": span["size"],
                            "font": span["font"],
                            "flags": span["flags"],
                            "page": page.number,
                            "x0": span["bbox"][0],
                            "x1": span["bbox"][2],
                            "y0": span["bbox"][1],
                            "width": page_width,
                            "ocr": False
                        })
        else:
            # Scanned/image-based page: use OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang=OCR_LANGS)
            n_boxes = len(ocr_data['level'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                size = ocr_data['height'][i] / 2
                spans.append({
                    "text": text,
                    "size": size,
                    "font": "OCR",
                    "flags": 0,
                    "page": page.number,
                    "x0": ocr_data['left'][i] / 2,
                    "x1": (ocr_data['left'][i] + ocr_data['width'][i]) / 2,
                    "y0": ocr_data['top'][i] / 2,
                    "width": page_width,
                    "ocr": True
                })
    return spans

def group_lines(spans):
    # Merges spans that are on the same line into a single line entry
    lines = []
    spans.sort(key=lambda s: (s["page"], s["y0"], s["x0"]))
    current_line = None
    for s in spans:
        if current_line and abs(current_line["y0"] - s["y0"]) < 2 and current_line["page"] == s["page"]:
            current_line["spans"].append(s)
        else:
            if current_line:
                lines.append(current_line)
            current_line = {"page": s["page"], "y0": s["y0"], "spans": [s]}
    if current_line:
        lines.append(current_line)

    output = []
    for line in lines:
        text = " ".join(s["text"] for s in line["spans"])
        size = np.mean([s["size"] for s in line["spans"]])
        alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
        width_frac = (max(s["x1"] for s in line["spans"]) - min(s["x0"] for s in line["spans"])) / line["spans"][0]["width"]
        x0 = min(s["x0"] for s in line["spans"])
        ocr = any(s.get("ocr", False) for s in line["spans"])
        trailing = " " if text and text[-1] == " " else ""
        output.append({
            "text": text.strip() + trailing,
            "size": round(size, 2),
            "font": line["spans"][0]["font"],
            "page": line["page"],
            "y0": line["y0"],
            "alpha": round(alpha, 2),
            "wf": round(width_frac, 2),
            "x0": round(x0, 2),
            "ocr": ocr
        })
    return output

def join_numbered_fields(lines):
    # Merges lines like "1.2" and "Introduction" into "1.2 Introduction"
    joined = []
    i = 0
    while i < len(lines):
        current = lines[i]["text"].strip()
        next_line = lines[i+1]["text"].strip() if i + 1 < len(lines) else ""
        if re.match(r'^\d+(\.\d+)*\.?$', current) and next_line and len(next_line.split()) <= 10:
            merged_text = current + " " + next_line
            joined.append({
                **lines[i],
                "text": merged_text,
                "size": max(lines[i]["size"], lines[i+1]["size"]),
            })
            i += 2
        else:
            joined.append(lines[i])
            i += 1
    return joined

def is_repeated_digit_line(text):
    # Skips headings like "10 10 10 10"
    tokens = text.strip().split()
    if len(tokens) >= 3 and all(t.isdigit() and t == tokens[0] for t in tokens):
        return True
    return False

def merge_split_headings(lines, font_size_threshold=1.0, vertical_threshold=32):
    # Merges consecutive lines that probably form a split heading (common in scanned PDFs)
    merged = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        merged_line = cur.copy()
        while (
            i+1 < len(lines)
            and lines[i+1]["page"] == merged_line["page"]
            and abs(lines[i+1]["y0"] - merged_line["y0"]) < vertical_threshold
            and abs(lines[i+1]["size"] - merged_line["size"]) < font_size_threshold
            and not merged_line["text"].endswith(".")   # Avoid merging paragraphs
            and len(lines[i+1]["text"].split()) < 8     # Only merge short heading parts
        ):
            # Merge text and use max font size
            merged_line["text"] = (merged_line["text"].rstrip() + " " + lines[i+1]["text"].lstrip()).strip()
            merged_line["size"] = max(merged_line["size"], lines[i+1]["size"])
            i += 1
        merged.append(merged_line)
        i += 1
    return merged
