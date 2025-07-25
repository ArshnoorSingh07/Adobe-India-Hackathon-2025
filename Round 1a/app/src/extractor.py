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
