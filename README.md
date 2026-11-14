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

All tests run offline. Embeddings currently come from an offline **mock** provider
(word hashing): it exercises the pipeline mechanics but does **not** validate semantic
retrieval quality. See [backend/tests/README.md](backend/tests/README.md) for what is
and isn't validated until a real embedding model is plugged in.

## Try the pipeline

```bash
cd backend
python -m app.services.ingestor path/to/file.pdf           # preview chunks, writes nothing
python -m app.services.ingestor path/to/file.pdf --index   # store in vectorstore/
```

Settings come from environment variables or a `.env` file (see `.env.example`).
Delete the contents of `vectorstore/` to clear the index.

## Status

- [x] Step 2: PDF loading and parsing (`backend/app/services/ingestor.py`)
- [x] Step 3: Chunking (`chunk_documents`, `load_and_chunk_pdf` in `ingestor.py`)
- [x] Step 4: Embeddings + ChromaDB (`index_pdf`; offline mock embeddings for now, see `backend/tests/README.md`)
- [ ] Retrieval
- [ ] LLM (Ollama + Llama 3)
- [ ] Streamlit UI
