from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Save MiniLM encoder to app/models
encoder = SentenceTransformer("all-MiniLM-L6-v2")
encoder.save("models/all-MiniLM-L6-v2")

# Save Flan-T5-small to app/models
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

tokenizer.save_pretrained("models/flan-t5-small/tokenizer")
model.save_pretrained("models/flan-t5-small/model")
