
from rag_pipeline import ask_question

# Hotel policy test questions
test_questions = [
    "Are service animals charged a pet fee?",
    "What is the cancellation deadline for flexible reservations?",
    "What are the hotel's quiet hours?",
    "Is an incidental hold the same as an actual charge?",
    "Does AuroraStays provide free airport transportation?"
]

print("=" * 70)
print("AURORASTAYS QnA - RAG TESTING")
print("=" * 70)

for number, question in enumerate(test_questions, start=1):

    print(f"\nTEST CASE {number}")
    print("-" * 70)

    print(f"Question: {question}")

    try:
        result = ask_question(question)

        print("\nGenerated Answer:")
        print(result["answer"])

        print("\nRetrieved Sources:")

        for source in result["sources"]:
            print(
                f"- {source['source']} "
                f"(Hybrid Score: {source['hybrid_score']})"
            )

    except Exception as error:
        print(f"TEST FAILED: {error}")

    print("=" * 70)

print("\nRAG TESTING COMPLETED!")
