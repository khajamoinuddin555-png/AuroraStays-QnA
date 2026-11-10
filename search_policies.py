
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project directories
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

INDEX_FILE = VECTORSTORE_DIR / "faiss.index"
METADATA_FILE = VECTORSTORE_DIR / "chunks_metadata.json"

# Embedding model used in Step 6
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of results to retrieve
TOP_K = 5


def load_retriever():

    print("Loading FAISS database...")

    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


def search_policies(question, index, metadata, model):

    # Convert user question into embedding
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    # Search FAISS database
    scores, indices = index.search(query_embedding, TOP_K)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        result = metadata[idx].copy()
        result["similarity_score"] = float(score)

        results.append(result)

    return results


if __name__ == "__main__":

    print("=" * 60)
    print("AURORASTAYS QnA - SEMANTIC POLICY SEARCH")
    print("=" * 60)

    index, metadata, model = load_retriever()

    print(f"\nDatabase vectors: {index.ntotal}")

    # Test question
    question = "What are the hotel's quiet hours?"

    print(f"\nUser Question: {question}")

    results = search_policies(
        question,
        index,
        metadata,
        model
    )

    print("\n" + "=" * 60)
    print("TOP RETRIEVED POLICY CHUNKS")
    print("=" * 60)

    for position, result in enumerate(results, start=1):

        print(f"\nResult {position}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Policy Text:\n{result['text']}")
        print("-" * 60)

    print("\nSemantic search completed successfully!")
