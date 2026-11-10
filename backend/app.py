
import sys
from pathlib import Path

from flask import Flask, request, jsonify


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# 2. IMPORT RAG PIPELINE
# ============================================================

from rag_pipeline import ask_question


# ============================================================
# 3. INITIALIZE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# 4. HEALTH CHECK API
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "success",
        "message": "AuroraStays QnA API is running",
        "service": "Flask Backend"
    }), 200


# ============================================================
# 5. QUESTION-ANSWERING API
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        # Read JSON request
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body must contain valid JSON."
            }), 400

        question = data.get("question")

        # Validate question
        if not isinstance(question, str) or not question.strip():

            return jsonify({
                "status": "error",
                "message": "Please provide a valid question."
            }), 400

        # Call RAG pipeline
        result = ask_question(question.strip())

        # Prepare relevant retrieved chunks
        relevant_chunks = []

        for chunk in result["retrieved_chunks"]:

            relevant_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "hybrid_score": round(
                    chunk["hybrid_score"], 4
                ),
                "text": chunk["text"]
            })

        # Return API response
        return jsonify({
            "status": "success",
            "question": question.strip(),
            "answer": result["answer"],
            "sources": result["sources"],
            "relevant_chunks": relevant_chunks
        }), 200

    except Exception as error:

        app.logger.exception("Question-answering API failed")

        return jsonify({
            "status": "error",
            "message": "Unable to process the question."
        }), 500


# ============================================================
# 6. RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AURORASTAYS QnA - FLASK BACKEND")
    print("=" * 60)

    print("\nStarting Flask API server...")
    print("Health URL: http://127.0.0.1:5000/health")
    print("Ask URL: http://127.0.0.1:5000/ask")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
