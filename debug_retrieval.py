
from hybrid_search import hybrid_search
from rag_pipeline import ask_question

questions = [
    "For ASH-SEA Pike, is breakfast included and what should I budget for parking?",
    "I booked AuroraStay Austin North. Is there a pool, and can I reserve a cabana?",
    "The room next door has been loud since midnight. What is AuroraStay supposed to do?",
    "I canceled my flexible booking after the deadline and now see a one-night charge. Can someone still refund it?"
]

for number, question in enumerate(questions, start=1):

    print("\n" + "=" * 80)
    print(f"QUESTION {number}")
    print("=" * 80)
    print(question)

    retrieved_chunks = hybrid_search(question, top_k=5)

    print("\nRETRIEVED CHUNKS:")

    for rank, chunk in enumerate(retrieved_chunks, start=1):

        print("\n" + "-" * 70)
        print(f"Rank: {rank}")
        print(f"Hybrid Score: {chunk['hybrid_score']:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("\nChunk Content:")
        print(chunk["text"])

    print("\nGENERATED RAG ANSWER:")

    result = ask_question(question)

    print(result["answer"])

print("\n" + "=" * 80)
print("RETRIEVAL DEBUGGING COMPLETED")
print("=" * 80)
