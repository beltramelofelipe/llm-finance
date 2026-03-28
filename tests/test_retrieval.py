"""Unit tests for RetrievalService."""

import pytest

from tests.conftest import EMBEDDING_DIM


def _populate_chunks(chunks_collection, n: int = 5):
    """Insert n dummy chunks into the collection."""
    ids = [f"doc1__chunk_{i:04d}" for i in range(n)]
    embeddings = [[float(i) / n] * EMBEDDING_DIM for i in range(n)]
    documents = [f"Chunk number {i} with some financial text." for i in range(n)]
    metadatas = [
        {
            "document_id": "doc1",
            "filename": "report.pdf",
            "page_number": 1,
            "chunk_index": i,
            "chunk_count": n,
            "page_count": 1,
            "uploaded_at": "2024-01-01T00:00:00+00:00",
            "size_bytes": 1024,
        }
        for i in range(n)
    ]
    chunks_collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


@pytest.mark.asyncio
async def test_retrieve_returns_top_k(retrieval_service, chroma_collections):
    chunks, _ = chroma_collections
    _populate_chunks(chunks, n=5)

    results = await retrieval_service.retrieve("What is the revenue?", top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_relevance_score_range(retrieval_service, chroma_collections):
    chunks, _ = chroma_collections
    _populate_chunks(chunks, n=5)

    results = await retrieval_service.retrieve("What is the revenue?", top_k=5)
    for r in results:
        assert 0.0 <= r["relevance_score"] <= 1.0


@pytest.mark.asyncio
async def test_retrieve_uses_default_top_k(retrieval_service, chroma_collections):
    chunks, _ = chroma_collections
    _populate_chunks(chunks, n=5)

    results = await retrieval_service.retrieve("revenue")
    # default top_k is 3 (set in fixture)
    assert len(results) <= 3
