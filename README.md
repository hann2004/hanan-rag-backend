# Hanan Nasir RAG Backend

This is the backend for Hanan Nasir's portfolio AI assistant. It is a FastAPI app that provides a Retrieval-Augmented Generation (RAG) API for answering questions about Hanan's projects, skills, and experience.

## Features
- FastAPI backend with a /rag-query endpoint
- Loads portfolio context from data/portfolio_context.txt
- Uses ChromaDB and sentence-transformers for semantic search
- Integrates with Groq LLM API for answer generation

## Requirements
- Python 3.9+
- See requirements.txt for Python dependencies

## Setup
1. Clone this repo:
   ```sh
   git clone https://github.com/yourusername/hanan-rag-backend.git
   cd hanan-rag-backend
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set your Groq API key as an environment variable:
   ```sh
   export GROQ_API_KEY=your_groq_api_key
   ```
4. Run the server:
   ```sh
   uvicorn rag_backend.main:app --host 0.0.0.0 --port 8000
   ```

## Deployment
- Recommended: Deploy on [Render](https://render.com/) as a Python web service.
- Set the build command: `pip install -r requirements.txt`
- Set the start command: `uvicorn rag_backend.main:app --host 0.0.0.0 --port 8000`
- Set the environment variable: `GROQ_API_KEY`

## Project Structure
```
.
├── rag_backend/
│   └── main.py
├── data/
│   ├── portfolio_context.txt
│   └── portfolio_docs.json
├── requirements.txt
└── README.md
```

## Notes
- The backend expects the data folder to be present at the project root.
- For production, set your frontend to use the backend's public URL.

---

MIT License