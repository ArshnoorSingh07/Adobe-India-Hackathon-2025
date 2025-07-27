from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# === Setup base path ===
base_path = os.path.dirname(__file__)  # This points to 'Round 1b/app'
models_path = os.path.join(base_path, "models")

# === Save MiniLM encoder to Round 1b/app/models/all-MiniLM-L6-v2 ===
encoder = SentenceTransformer("all-MiniLM-L6-v2")
minilm_path = os.path.join(models_path, "all-MiniLM-L6-v2")
encoder.save(minilm_path)

# === Save Flan-T5-small to Round 1b/app/models/flan-t5-small ===
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

flan_t5_path = os.path.join(models_path, "flan-t5-small")
tokenizer.save_pretrained(os.path.join(flan_t5_path, "tokenizer"))
model.save_pretrained(os.path.join(flan_t5_path, "model"))
