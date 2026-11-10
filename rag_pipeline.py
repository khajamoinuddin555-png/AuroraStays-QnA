
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from hybrid_search import hybrid_search


# ============================================================
# 1. LOAD ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOP_K = 5

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env")

if not BASE_URL:
    raise ValueError("OPENAI_BASE_URL is missing in .env")


# ============================================================
# 2. INITIALIZE GREAT LEARNING LLM
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

print("Loading AuroraStays hybrid retrieval system...")
print("Hybrid retriever initialized successfully.")


# ============================================================
# 3. RAG QUESTION-ANSWERING FUNCTION
# ============================================================

def ask_question(question):

    if not question or not question.strip():
        raise ValueError("Please enter a valid question.")

    # Retrieve relevant policy chunks
    retrieved_chunks = hybrid_search(
        question,
        top_k=TOP_K
    )

    if not retrieved_chunks:
        return {
            "answer": (
                "I could not find relevant information "
                "in the available hotel policies."
            ),
            "sources": [],
            "retrieved_chunks": []
        }

    # Prepare policy context
    context_parts = []

    for position, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
Source {position}: {chunk['source']}
Chunk ID: {chunk['chunk_id']}

Policy Content:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)


    # ========================================================
    # 4. SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are AuroraStays QnA, an internal hotel policy assistant.

Your responsibility is to answer hotel staff questions
using ONLY the supplied AuroraStays policy context.

STRICT RULES:

1. Use only the supplied policy context.
   Never invent or assume policy information.

2. Preserve important policy conditions, exceptions,
   amounts, time limits, and approval requirements.

3. Distinguish between the normal policy rule and
   decisions that require individual review.

4. LATE CANCELLATION AND REFUND QUESTIONS:

   - Explain that a late cancellation or no-show may
     result in one night's charge plus applicable tax
     being retained.

   - Do not describe this as an absolute prohibition
     on refunds unless the policy explicitly says so.

   - If the context states that a manager may approve
     a partial refund, credit, or another remedy based
     on documented impact, explain this possibility.

   - Never promise that a manager will approve a refund.

   - Mention that the case requires review and approval
     according to the supplied policy.

5. When the context does not contain enough information
   to decide an individual case, explain what is known
   and what needs verification.

6. Do not contradict an answer by first providing a
   policy rule and then unnecessarily saying that the
   entire answer is unavailable.

7. For property-specific questions, do not assume that
   every property provides the same amenities or services.
   Use the supplied property information only.

8. If the requested property information is not available
   in the supplied context, clearly explain this limitation
   and recommend checking the live property profile or
   contacting the property team.

9. Cite the relevant source document names in your answer.
   Mention only documents that support the answer.

10. Keep answers clear, concise, professional, and useful
    for hotel staff.
"""


    # ========================================================
    # 5. GENERATE ANSWER USING GREAT LEARNING LLM
    # ========================================================

    user_prompt = f"""
HOTEL POLICY CONTEXT:

{context}

STAFF QUESTION:

{question}

Answer the staff question using only the supplied
hotel policy context.

Explain the applicable policy, relevant exceptions,
and any required approval or verification.

Do not make unsupported promises.
"""


    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("The LLM returned an empty answer.")


    # ========================================================
    # 6. COLLECT SUPPORTING SOURCE INFORMATION
    # ========================================================

    sources = []

    for chunk in retrieved_chunks:

        sources.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "hybrid_score": round(
                chunk["hybrid_score"],
                4
            )
        })


    # ========================================================
    # 7. RETURN FINAL RAG RESPONSE
    # ========================================================

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks
    }


# ============================================================
# 8. TEST RAG PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("AURORASTAYS QnA - RAG PIPELINE")
    print("=" * 60)

    question = (
        "I canceled my flexible booking after the deadline "
        "and now see a one-night charge. Can someone still "
        "refund it?"
    )

    print("\nStaff Question:")
    print(question)

    result = ask_question(question)

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("SUPPORTING RETRIEVED SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print("\nDocument:", source["source"])
        print("Chunk ID:", source["chunk_id"])
        print("Hybrid Score:", source["hybrid_score"])

    print("\n" + "=" * 60)
    print("RAG PIPELINE COMPLETED")
    print("=" * 60)
