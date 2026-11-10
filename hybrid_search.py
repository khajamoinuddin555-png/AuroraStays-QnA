
import json
import re
import faiss
import numpy as np

from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. LOAD FAISS VECTOR DATABASE AND CHUNK METADATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

index = faiss.read_index(
    str(VECTORSTORE_DIR / "faiss.index")
)

with open(
    VECTORSTORE_DIR / "chunks_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. STOP WORDS
# ============================================================

STOP_WORDS = {
    "what", "are", "the", "is", "a", "an", "of", "to",
    "and", "in", "on", "for", "does", "do", "when",
    "how", "can", "i", "we", "hotel", "guest"
}


# ============================================================
# 3. TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ============================================================
# 4. QUERY EXPANSION
# ============================================================

def expand_query(question):

    query = question.lower()
    additional_terms = []

    # Noise and disturbance questions
    noise_keywords = [
        "loud", "noise", "noisy", "disturbance",
        "next door", "midnight"
    ]

    if any(word in query for word in noise_keywords):

        additional_terms.append(
            "quiet hours noise disturb others "
            "contact the room document complaint "
            "escalate repeat serious disturbances "
            "manager on duty"
        )

    # Late cancellation and refund questions
    refund_keywords = [
        "late cancellation", "after deadline",
        "cancelled after", "canceled after",
        "late cancel", "refund"
    ]

    if any(word in query for word in refund_keywords):

        additional_terms.append(
            "late cancellation no-show one night "
            "applicable tax may be retained "
            "approved exception manager approval "
            "service recovery manager may approve "
            "partial refund credit or other remedy "
            "based on documented impact "
            "complaint facts remedy offered "
            "documented system error duplicate booking"
        )

    # Pool and cabana questions
    amenity_keywords = [
        "pool", "cabana"
    ]

    if any(word in query for word in amenity_keywords):

        additional_terms.append(
            "property amenities pool availability "
            "live property profile hours reservations "
            "closures property team"
        )

    if additional_terms:
        return question + " " + " ".join(additional_terms)

    return question


# ============================================================
# 5. BM25 KEYWORD SEARCH
# ============================================================

def bm25_search(question, documents):

    query_terms = [
        word for word in tokenize(question)
        if word not in STOP_WORDS
    ]

    tokenized_docs = [
        tokenize(item["text"])
        for item in documents
    ]

    doc_count = len(tokenized_docs)

    avg_length = (
        sum(len(doc) for doc in tokenized_docs) / doc_count
    )

    document_frequency = Counter()

    for doc in tokenized_docs:
        document_frequency.update(set(doc))

    scores = []

    for doc in tokenized_docs:

        frequencies = Counter(doc)
        score = 0.0

        for term in query_terms:

            if term not in frequencies:
                continue

            df = document_frequency[term]

            idf = np.log(
                1 + (doc_count - df + 0.5) / (df + 0.5)
            )

            tf = frequencies[term]
            doc_length = len(doc)

            k1 = 1.5
            b = 0.75

            denominator = tf + k1 * (
                1 - b + b * doc_length / avg_length
            )

            score += idf * (
                tf * (k1 + 1)
            ) / denominator

        scores.append(score)

    return np.argsort(scores)[::-1], scores


# ============================================================
# 6. HYBRID SEARCH
# ============================================================

def hybrid_search(question, top_k=5):

    if not question or not question.strip():
        raise ValueError("Please enter a valid search question.")

    # Expand the original question
    expanded_question = expand_query(question)

    # Generate query embedding
    query_embedding = model.encode(
        [expanded_question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    # Semantic search using FAISS
    semantic_scores, _ = index.search(
        query_embedding,
        index.ntotal
    )

    # Keyword search using BM25
    _, lexical_scores = bm25_search(
        expanded_question,
        metadata
    )

    semantic = semantic_scores[0]

    lexical = np.array(
        lexical_scores,
        dtype=np.float32
    )

    # Normalize semantic scores
    semantic_norm = (
        (semantic - semantic.min()) /
        (semantic.max() - semantic.min() + 1e-10)
    )

    # Normalize lexical scores
    lexical_norm = (
        (lexical - lexical.min()) /
        (lexical.max() - lexical.min() + 1e-10)
    )

    # Combine lexical and semantic scores
    combined_scores = (
        0.8 * lexical_norm +
        0.2 * semantic_norm
    )

    # Rank documents by combined score
    ranked_indices = np.argsort(
        combined_scores
    )[::-1]

    results = []

    for idx in ranked_indices[:top_k]:

        result = metadata[int(idx)].copy()

        result["hybrid_score"] = float(
            combined_scores[idx]
        )

        results.append(result)

    return results


# ============================================================
# 7. TEST HYBRID SEARCH
# ============================================================

if __name__ == "__main__":

    question = "What are the hotel's quiet hours?"

    print("=" * 70)
    print("AURORASTAYS QnA - HYBRID SEARCH")
    print("=" * 70)

    print("\nQuestion:", question)

    results = hybrid_search(question)

    for position, result in enumerate(results, start=1):

        print(f"\nResult {position}")

        print("Chunk:", result["chunk_id"])

        print("Source:", result["source"])

        print(
            "Hybrid Score:",
            round(result["hybrid_score"], 5)
        )

        print("Policy Text:", result["text"])

    print("\nHybrid search completed!")
