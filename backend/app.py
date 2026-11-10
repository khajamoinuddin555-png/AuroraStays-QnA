
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
from hybrid_search import hybrid_search


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
# 7. FULL RAG INFERENCE API
# ============================================================

@app.route("/v1/answer_with_relevant_chunks", methods=["POST"])
def answer_with_relevant_chunks():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body must contain valid JSON."
            }), 400

        query = data.get("query")
        k = data.get("k", 3)
        model_name = data.get("model_name")
        temperature = data.get("temperature", 0)
        top_p = data.get("top_p", 1)
        max_tokens = data.get("max_tokens", 500)

        if not isinstance(query, str) or not query.strip():
            return jsonify({
                "status": "error",
                "message": "Please provide a valid query."
            }), 400

        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            return jsonify({
                "status": "error",
                "message": "k must be a positive integer."
            }), 400

        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
            return jsonify({
                "status": "error",
                "message": "temperature must be between 0 and 2."
            }), 400

        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not 0 < top_p <= 1:
            return jsonify({
                "status": "error",
                "message": "top_p must be greater than 0 and at most 1."
            }), 400

        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            return jsonify({
                "status": "error",
                "message": "max_tokens must be a positive integer."
            }), 400

        result = ask_question(
            query.strip(),
            top_k=k,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        relevant_chunks = []

        for chunk in result["retrieved_chunks"]:

            relevant_chunks.append({
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "hybrid_score": round(
                        chunk["hybrid_score"], 4
                    )
                }
            })

        return jsonify({
            "status": "success",
            "query": query.strip(),
            "answer": result["answer"],
            "chunks": relevant_chunks,
            "parameters": {
                "k": k,
                "model_name": model_name or "configured default",
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }
        }), 200

    except Exception:

        app.logger.exception("Full RAG inference API failed")

        return jsonify({
            "status": "error",
            "message": "Unable to generate the answer."
        }), 500
        
# ============================================================
# 6. RUN FLASK SERVER
# ============================================================

@app.route("/v1/relevant_chunks", methods=["POST"])
def get_relevant_chunks():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body must contain valid JSON."
            }), 400

        query = data.get("query")
        k = data.get("k", 3)

        if not isinstance(query, str) or not query.strip():
            return jsonify({
                "status": "error",
                "message": "Please provide a valid query."
            }), 400

        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            return jsonify({
                "status": "error",
                "message": "k must be a positive integer."
            }), 400

        chunks = hybrid_search(
            query.strip(),
            top_k=k
        )

        return jsonify({
            "status": "success",
            "query": query.strip(),
            "k": k,
            "chunks": [
                {
                    "text": chunk["text"],
                    "metadata": {
                        "chunk_id": chunk["chunk_id"],
                        "source": chunk["source"],
                        "hybrid_score": round(
                            chunk["hybrid_score"], 4
                        )
                    }
                }
                for chunk in chunks
            ]
        }), 200

    except Exception:

        app.logger.exception("Retrieval API failed")

        return jsonify({
            "status": "error",
            "message": "Unable to retrieve relevant chunks."
        }), 500


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
