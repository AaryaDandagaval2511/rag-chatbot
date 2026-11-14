# RAG Conversational Chatbot

Chat with your PDFs: upload documents and ask questions answered from their content,
using a local LLM (Llama 3 via Ollama), ChromaDB, LangChain, FastAPI and Streamlit.

> Work in progress — built incrementally.

## Local setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
```

## Run tests

```bash
cd backend && pytest -v
```

## Status

- [x] Step 2: PDF loading and parsing (`backend/app/services/ingestor.py`)
- [x] Step 3: Chunking (`chunk_documents`, `load_and_chunk_pdf` in `ingestor.py`)
- [ ] Embeddings + ChromaDB
- [ ] Retrieval
- [ ] LLM (Ollama + Llama 3)
- [ ] Streamlit UI
