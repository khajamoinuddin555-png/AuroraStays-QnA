# AuroraStays-QnA
### AI-Powered Hotel Policy Question Answering System using Retrieval-Augmented Generation (RAG)

## 1. Project Overview

AuroraStays-QnA is an internal AI-powered question-answering system designed to help hotel staff retrieve accurate information from hotel policy documents.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant policy information and generate context-grounded answers using a Large Language Model (LLM).

## 2. Business Problem

Hotel staff handle approximately 1,200 queries per week. Finding information across multiple policy documents takes significant time and can lead to inconsistent answers.

This project aims to:
- Simplify hotel policy information retrieval.
- Provide reliable, context-grounded answers.
- Display supporting document sources.
- Reduce manual policy lookup and unnecessary escalation.

## 3. Key Features

- Natural-language question answering.
- PDF policy document processing.
- Semantic and keyword-based hybrid retrieval.
- FAISS vector database.
- LLM-powered answer generation.
- Supporting sources and retrieved chunks.
- Evaluation using four RAG metrics.
- Flask REST API backend.
- Streamlit user interface.
- Docker-based deployment.

## 4. System Architecture

1. Upload hotel policy PDF documents.
2. Extract text using Docling.
3. Split documents into overlapping chunks.
4. Generate embeddings using Sentence Transformers.
5. Store embeddings in a FAISS vector database.
6. Retrieve relevant chunks using hybrid search.
7. Send retrieved context to the LLM.
8. Generate answers grounded in retrieved policies.
9. Display answers and supporting sources through Streamlit.

## 5. Technologies Used

| Component | Technology |
|---|---|
| Programming | Python 3.11 |
| PDF Processing | Docling |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Retrieval | BM25 + Semantic Search |
| LLM | Great Learning API - gpt-4o-mini |
| RAG Framework | LangChain components |
| Evaluation | DeepEval |
| Backend | Flask + Gunicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Version Control | Git + GitHub |

## 6. Project Structure

```text
AuroraStays-QnA/
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   ├── embeddings/
│   ├── evaluation/
│   └── evaluation_results/
├── vectorstore/
│   ├── faiss.index
│   └── chunks_metadata.json
├── extract_pdfs.py
├── chunk_documents.py
├── generate_embeddings.py
├── build_vectorstore.py
├── hybrid_search.py
├── search_policies.py
├── rag_pipeline.py
├── evaluation_llm.py
├── evaluate_rag.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## 7. Installation and Configuration

### Prerequisites

- Python 3.11
- Git
- Docker Desktop
- Great Learning API key

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AuroraStays-QnA
```

### Create virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file using `.env.example` as a template.

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://aibe.mygreatlearning.com/openai/v1
OPENAI_MODEL=gpt-4o-mini
```
**API Configuration**

This project was developed and evaluated using the Great Learning OpenAI-compatible API endpoint with the `gpt-4o-mini` model.

Users or reviewers using another OpenAI-compatible provider can update the `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` values in their `.env` file according to their provider's configuration.

**Security Note:** Never commit your actual API key or `.env` file to GitHub. Use `.env.example` as a configuration template.

Never upload your actual `.env` file or API key to GitHub.

## 8. Run the Application Locally

### Start Flask backend

```bash
python backend/app.py
```

Backend URL:

http://localhost:5000

### Start Streamlit frontend

Open another terminal and activate the virtual environment.

```bash
streamlit run frontend/app.py --server.port 8501
```

Frontend URL:

http://localhost:8501

## 9. Docker Deployment

Make sure Docker Desktop is running.

### Build backend image

```bash
docker build -t aurorastays-backend -f backend/Dockerfile .
```

### Start both services

```bash
docker compose up -d --build
```

### Check containers

```bash
docker ps
```

### Access application

Frontend: http://localhost:8501

Backend: http://localhost:7860

### Stop containers

```bash
docker compose down
```

## 10. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Backend health check |
| POST | /ask | Submit a hotel policy question |

Example request:

```json
{
  "question": "What are the quiet hours at AuroraStay hotels?"
}
```

The API returns the generated answer, supporting sources, and relevant retrieved chunks.

## 11. RAG Evaluation

The system was evaluated using two datasets:
- Evaluation dataset: 10 questions.
- Held-out test dataset: 10 questions.

Four DeepEval metrics were used.

### Improved Evaluation Results

| Metric | Evaluation | Held-out Test |
|---|---:|---:|
| Faithfulness | 87.05% | 85.78% |
| Contextual Recall | 92.50% | 91.67% |
| Contextual Precision | 80.00% | 74.26% |
| Answer Relevancy | 89.19% | 87.83% |
| **Average** | **87.19%** | **84.89%** |

The evaluation scores are LLM-based estimates and should be complemented by manual review of generated answers and retrieved sources.

### Run evaluation

```bash
python evaluate_rag.py evaluation
```

```bash
python evaluate_rag.py test
```

Results are saved under `data/evaluation_results/`.

## 12. Testing

The following functionality was tested:

- PDF extraction and document processing.
- Chunk creation and embedding generation.
- FAISS vector database creation.
- Semantic and hybrid retrieval.
- RAG answer generation.
- Abstention when relevant information is unavailable.
- Flask health and question-answering endpoints.
- Streamlit frontend.
- Docker backend and frontend containers.
- End-to-end question answering with supporting sources.

## 13. Future Enhancements

- Add more hotel properties and policy documents.
- Implement user authentication and role-based access.
- Improve retrieval precision and contextual ranking.
- Add conversation history.
- Implement production monitoring and logging.
- Deploy to a cloud platform.
- Expand the evaluation dataset.

## 14. Conclusion

AuroraStays-QnA demonstrates how Retrieval-Augmented Generation can be used to build an internal hotel policy question-answering application.

By combining document processing, hybrid retrieval, vector search, and LLM-based generation, the system provides staff with relevant answers and supporting policy sources through an easy-to-use interface.

---

**Project:** AuroraStays-QnA  
**Category:** Generative AI | Retrieval-Augmented Generation | NLP  
**Deployment:** Docker  
**Interface:** Streamlit  
**Backend:** Flask