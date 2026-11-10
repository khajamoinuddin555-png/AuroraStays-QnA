
import json
import faiss
import numpy as np

from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_FILE = BASE_DIR / "data" / "embeddings" / "embeddings.npy"
METADATA_FILE = BASE_DIR / "data" / "embeddings" / "chunks_metadata.json"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Create vector database directory
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AURORASTAYS QnA - FAISS VECTOR DATABASE")
print("=" * 60)

# Load embeddings
embeddings = np.load(EMBEDDING_FILE)

# Ensure FAISS-compatible format
embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

# Load metadata
with METADATA_FILE.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

# Validate data
if len(embeddings) != len(metadata):
    raise ValueError("Embedding count and metadata count do not match")

if len(embeddings) == 0:
    raise ValueError("No embeddings available")

print(f"\nTotal embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

# Create FAISS index
dimension = embeddings.shape[1]

# Inner product works as cosine similarity because
# our embeddings were normalized during generation.
index = faiss.IndexFlatIP(dimension)

# Add vectors to FAISS
index.add(embeddings)

# Save FAISS index
index_file = VECTORSTORE_DIR / "faiss.index"

faiss.write_index(index, str(index_file))

# Save corresponding metadata
metadata_file = VECTORSTORE_DIR / "chunks_metadata.json"

with metadata_file.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

# Verification
loaded_index = faiss.read_index(str(index_file))

print("\n" + "=" * 60)
print("FAISS DATABASE SUMMARY")
print("=" * 60)

print(f"Total vectors stored: {loaded_index.ntotal}")
print(f"Vector dimensions: {loaded_index.d}")
print(f"Index type: {type(loaded_index).__name__}")
print(f"Index saved: {index_file}")
print(f"Metadata saved: {metadata_file}")

if loaded_index.ntotal != len(metadata):
    raise RuntimeError("FAISS verification failed")

print("\nFAISS vector database created successfully!")
