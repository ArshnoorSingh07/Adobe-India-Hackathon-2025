import os
import re
import json
import unicodedata
from datetime import datetime
from collections import defaultdict

import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'all-MiniLM-L6-v2')

TOP_K = 5  # Number of top sections to output per collection
TFIDF_WEIGHT = 0.2
HDR_WEIGHT   = 0.3
EMB_WEIGHT   = 0.5

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

encoder = SentenceTransformer(MODEL_PATH)

def load_rewriter():
    """
    Load a local Flan-T5 summarizer model if available.
    Falls back to None if not found (no online downloads).
    """
    for model_name in ("google/flan-t5-base", "google/flan-t5-small"):
        try:
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, local_files_only=True, torch_dtype="float32"
            )
            return pipeline(
                "text2text-generation",
                model=mdl,
                tokenizer=tok,
                max_length=200,
                num_beams=4,
                do_sample=False,
                device=-1
            )
        except OSError:
            continue
    return None

rewriter = load_rewriter()

# === Persona/job metadata for each collection ===
COLLECTION_METADATA = {
    "Collection 1": ("Travel Planner", "Plan a trip of 4 days for a group of 10 college friends."),
    "Collection 2": ("HR professional", "Create and manage fillable forms for onboarding and compliance."),
    "Collection 3": ("Food Contractor", "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items.")
}

def clean_text(txt: str) -> str:
    """Normalize and clean up extracted text."""
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"[•▪·]", "", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def is_heading(txt: str) -> bool:
    """
    Heuristic: Is this line a heading?
    Heading = 2-12 words, title-case or uppercase, not a known footer/page number/etc.
    """
    words = txt.split()
    return (
        1 < len(words) <= 12
        and (txt.istitle() or txt.isupper())
        and not re.search(r"\b(page|copyright|table of contents|instructions)\b|\d{1,2}/\d{1,2}", txt.lower())
        and not txt.lower().endswith(":")
    )

def extract_sections(pdf_path):
    """
    Extract candidate sections as:
    [heading] + all lines until next heading,
    skipping short blocks.
    """
    doc = fitz.open(pdf_path)
    lines = []
    for pg, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0: continue
            for line in block.get("lines", []):
                texts = [clean_text(span["text"]) for span in line.get("spans", []) if clean_text(span["text"])]
                sizes = [span["size"] for span in line.get("spans", []) if clean_text(span["text"])]
                if texts and sizes:
                    lines.append({"text": " ".join(texts), "size": max(sizes), "page": pg})
    sec_map = {}
    i = 0
    while i < len(lines):
        l = lines[i]
        if is_heading(l["text"]):
            heading = l["text"]
            page   = l["page"]
            content_lines = []
            j = i + 1
            while j < len(lines) and not is_heading(lines[j]["text"]):
                content_lines.append(lines[j]["text"])
                j += 1
            section_content = " ".join(content_lines)
            # Only keep as section if long enough
            if len(section_content.split()) >= 15:
                sec_map[(heading, page)] = [section_content]
            i = j
        else:
            i += 1
    return sec_map

def summarize(text):
    """
    Summarize a section block using T5 if available, truncate long sections for safety.
    Fallback to the original text if the LLM output is too short or repeated.
    """
    max_words = 350
    words = text.split()
    if len(words) > max_words:
        text_to_summarize = " ".join(words[:max_words])
    else:
        text_to_summarize = text
    # Hard cap on input length for model safety
    if len(text_to_summarize) > 1800:
        text_to_summarize = text_to_summarize[:1800]
    if rewriter is not None:
        try:
            r1 = rewriter(f"rewrite for clarity: {text_to_summarize}")[0]["generated_text"]
            r2 = rewriter(f"correct grammar and clarity: {r1}")[0]["generated_text"]
            out = clean_text(r2)
            if out.lower().count("rewrite for clarity") > 1 or len(set(out.split())) <= 2 or len(out) < 30:
                return text_to_summarize
            return out
        except Exception:
            return text_to_summarize
    else:
        return text_to_summarize

def select_top_k_strictly_diverse(candidates, top_k=5):
    """
    Select top sections, enforcing at most one per PDF (maximum file diversity).
    """
    used_files = set()
    results = []
    for c in candidates:
        f = c["document"]
        if f in used_files:
            continue
        used_files.add(f)
        results.append(c)
        if len(results) == top_k:
            break
    return results

def process_collection(collection, persona, job):
    """
    For each collection, build all candidates and score/rank them.
    """
    query = f"{persona}: {job}"
    cands = []
    coll_dir = os.path.join(INPUT_DIR, collection, "PDFs")
    docs = sorted(f for f in os.listdir(coll_dir) if f.lower().endswith(".pdf"))
    for fn in docs:
        path = os.path.join(coll_dir, fn)
        for (hdr, pg), paras in extract_sections(path).items():
            merged = paras[0] if paras else ""
            if not merged or len(merged.split()) < 15:
                continue
            cands.append({
                "document": fn,
                "heading": hdr,
                "page": pg,
                "paras": paras,
                "merged": merged
            })
    if not cands:
        return docs, []
    # Build ranking features
    tfidf = TfidfVectorizer(stop_words="english")
    corpus = [query] + [c["merged"] for c in cands]
    mat = tfidf.fit_transform(corpus)
    q_vec = mat[0]
    doc_vecs = mat[1:]
    tfidf_scores = cosine_similarity(q_vec, doc_vecs)[0]
    q_emb = encoder.encode(query)
    for idx, c in enumerate(cands):
        hdr_emb = encoder.encode(c["heading"])
        body_emb = encoder.encode(c["merged"])
        emb_score = cosine_similarity([q_emb], [body_emb])[0][0]
        hdr_score = cosine_similarity([q_emb], [hdr_emb])[0][0]
        c["score"] = (
            EMB_WEIGHT   * emb_score +
            HDR_WEIGHT   * hdr_score +
            TFIDF_WEIGHT * tfidf_scores[idx]
        )
    cands.sort(key=lambda x: x["score"], reverse=True)
    top5 = select_top_k_strictly_diverse(cands, TOP_K)
    return docs, top5

def main():
    """
    Main batch pipeline: process each collection, output leaderboard JSON.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for collection, (persona, job) in COLLECTION_METADATA.items():
        print(f"Processing: {collection}")
        docs, top5 = process_collection(collection, persona, job)
        output = {
            "metadata": {
                "input_documents": docs,
                "persona": persona,
                "job_to_be_done": job.rstrip(".") + ".",
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        for rank, sec in enumerate(top5, start=1):
            raw = sec["merged"]
            refined = summarize(raw)
            output["extracted_sections"].append({
                "document":       sec["document"],
                "section_title":  sec["heading"],
                "importance_rank": rank,
                "page_number":    sec["page"]
            })
            output["subsection_analysis"].append({
                "document":     sec["document"],
                "refined_text": refined,
                "page_number":   sec["page"]
            })
        out_path = os.path.join(OUTPUT_DIR, f"{collection}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print("Output saved to", out_path)

if __name__ == "__main__":
    main()
