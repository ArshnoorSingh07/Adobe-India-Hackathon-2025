from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

base_path = os.path.dirname(__file__)
models_path = os.path.join(base_path, "models")

# Save MiniLM encoder
minilm_path = os.path.join(models_path, "all-MiniLM-L6-v2")
os.makedirs(minilm_path, exist_ok=True)
print("Downloading and saving all-MiniLM-L6-v2 ...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
encoder.save(minilm_path)
print("MiniLM model saved.")

# Save Flan-T5-small
flan_t5_path = os.path.join(models_path, "flan-t5-small")
os.makedirs(os.path.join(flan_t5_path, "model"), exist_ok=True)
os.makedirs(os.path.join(flan_t5_path, "tokenizer"), exist_ok=True)
print("Downloading and saving Flan-T5-small ...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer.save_pretrained(os.path.join(flan_t5_path, "tokenizer"))
model.save_pretrained(os.path.join(flan_t5_path, "model"))
print("Flan-T5-small model and tokenizer saved.")
