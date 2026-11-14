# Tests

Run from `backend/`:

```bash
pytest -v
```

All tests run **offline**: no Ollama, no downloaded models, no external APIs. `conftest.py`
blocks network connections for every test, so accidental internet access fails loudly.
Vector store tests use a temporary ChromaDB folder; your real `vectorstore/` is never touched.

## Two kinds of tests

### 1. Pipeline mechanics (all current tests)

These check that the code does what it says, using the **mock** embedding provider
`HashEmbeddings` (word hashing, see `app/services/embeddings.py`):

| File | Covers |
|---|---|
| `test_ingestor.py` | PDF loading, metadata, error handling, Unicode normalization |
| `test_chunking.py` | Chunk sizes, overlap, boundaries, metadata, `start_index` |
| `test_config.py` | Settings defaults, environment overrides, validation |
| `test_embeddings.py` | Mock embeddings: determinism, dimensions, unit length, interface |
| `test_vectorstore.py` | ChromaDB: cosine metric, model-identity check, upsert/delete, batching, no built-in embedding function |
| `test_indexing.py` | Deterministic IDs, duplicate skipping, updated files, changed settings, failure safety |

`HashEmbeddings` is **lexical, not semantic**: texts are similar only if they share words.
`test_mock_is_not_semantic` makes that limitation explicit. Passing these tests means the
pipeline *mechanics* are correct. It says **nothing** about retrieval *quality*.

### 2. Semantic tests (`@pytest.mark.semantic`, none yet)

Tests whose assertions only make sense with a real embedding model are marked
`semantic` and skip themselves while the mock provider is configured. They arrive with
retrieval in Step 5.

## Not validated until a real embedding model is plugged in

The mock cannot meaningfully test these. They remain **unverified** today:

- **Semantic retrieval quality**: finding a chunk that means the same thing with different
  words (e.g. "How many vacation days do I get?" -> "20 days of paid leave").
- **Parameter tuning**: whether chunk size / overlap, cosine distance, or any relevance
  threshold actually improve answers. These only mean something with real vectors.
- **Provider-specific behaviour**: nomic-embed-text's `search_document:` / `search_query:`
  prefixes, Ollama connection errors, model-not-downloaded errors, timeouts, batch speed.
- **Real vector sizes** (e.g. 768 dimensions): storage size and indexing speed.
- **Model-mismatch guard with real models**: the mechanism is tested with two mock
  identities, but not yet with real model names.
