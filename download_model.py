from sentence_transformers import SentenceTransformer

# Download model from HuggingFace and save locally
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("models/all-MiniLM-L6-v2")

print("✅ Model saved locally at: models/all-MiniLM-L6-v2")
