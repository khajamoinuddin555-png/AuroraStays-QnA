
import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "embeddings"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("=" * 60)
print("AURORASTAYS QnA - TEXT EMBEDDINGS")
print("=" * 60)

# Load chunks
chunks = []

with INPUT_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            chunks.append(json.loads(line))

if not chunks:
    raise ValueError("No chunks found in chunks.jsonl")

print(f"\nTotal chunks loaded: {len(chunks)}")

# Extract text
texts = [chunk["text"] for chunk in chunks]

# Load embedding model
print(f"\nLoading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

# Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Save embeddings
embedding_file = OUTPUT_DIR / "embeddings.npy"

np.save(embedding_file, embeddings)

# Save corresponding chunk metadata and text
metadata_file = OUTPUT_DIR / "chunks_metadata.json"

with metadata_file.open("w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4, ensure_ascii=False)

# Summary
print("\n" + "=" * 60)
print("EMBEDDING GENERATION SUMMARY")
print("=" * 60)

print(f"Total chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dimensions: {embeddings.shape[1]}")
print(f"Embeddings saved: {embedding_file}")
print(f"Metadata saved: {metadata_file}")

print("\nEmbeddings generated successfully!")
