"""Plain test helpers shared across test files (importable, unlike conftest.py)."""

from app.services.embeddings import HashEmbeddings

TEST_DIMENSIONS = 64  # small vectors keep tests fast; the mechanics are the same


class SpyEmbeddings(HashEmbeddings):
    """HashEmbeddings that records how many times, and on how many texts, it was called."""

    def __init__(self, dimensions=TEST_DIMENSIONS):
        super().__init__(dimensions)
        self.calls = 0
        self.texts_embedded = 0

    def embed_documents(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        return super().embed_documents(texts)
