
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent

index = faiss.read_index(str(BASE_DIR / "vectorstore" / "faiss.index"))

with open(BASE_DIR / "vectorstore" / "chunks_metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

questions = [
    "What are the hotel's quiet hours?",
    "When must guests keep noise levels low?",
    "What time does the hotel quiet period start and end?",
    "Quiet hours are 10:00 PM to 7:00 AM local property time."
]

for question in questions:
    embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embedding = np.ascontiguousarray(embedding, dtype=np.float32)

    scores, indices = index.search(embedding, 5)

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx]

        print(
            f"\nScore: {score:.4f}"
            f"\nChunk: {item['chunk_id']}"
            f"\nSource: {item['source']}"
        )
