"""Integration tests for the query endpoint."""

import io
from unittest.mock import patch

import pytest

from tests.conftest import EMBEDDING_DIM, FIXED_ANSWER


def _insert_chunks(chunks_collection, n: int = 3):
    """Pre-populate ChromaDB chunks so retrieval returns results."""
    ids = [f"doc1__chunk_{i:04d}" for i in range(n)]
    embeddings = [[0.0] * EMBEDDING_DIM for _ in range(n)]
    documents = [f"Revenue in quarter {i} was R$ {i * 3} billion." for i in range(n)]
    metadatas = [
        {
            "document_id": "doc1",
            "filename": "report.pdf",
            "page_number": 1,
            "chunk_index": i,
            "chunk_count": n,
            "page_count": 2,
            "uploaded_at": "2024-01-01T00:00:00+00:00",
            "size_bytes": 2048,
        }
        for i in range(n)
    ]
    chunks_collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


@pytest.mark.asyncio
async def test_query_returns_answer_and_sources(async_client, chroma_collections):
    chunks, _ = chroma_collections
    _insert_chunks(chunks, n=3)

    response = await async_client.post("/query", json={"question": "What was the quarterly revenue?"})
    assert response.status_code == 200
    body = response.json()

    assert body["question"] == "What was the quarterly revenue?"
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert isinstance(body["sources"], list)
    assert body["total_chunks_retrieved"] >= 1
    assert "model" in body


@pytest.mark.asyncio
async def test_query_source_structure(async_client, chroma_collections):
    chunks, _ = chroma_collections
    _insert_chunks(chunks, n=2)

    response = await async_client.post("/query", json={"question": "Show me the revenue figures."})
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) >= 1

    for source in sources:
        assert "document_id" in source
        assert "filename" in source
        assert "page_number" in source
        assert "chunk_index" in source
        assert "relevance_score" in source
        assert "excerpt" in source


@pytest.mark.asyncio
async def test_query_empty_question_422(async_client):
    response = await async_client.post("/query", json={"question": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_short_question_422(async_client):
    response = await async_client.post("/query", json={"question": "ab"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_custom_top_k(async_client, chroma_collections):
    chunks, _ = chroma_collections
    _insert_chunks(chunks, n=5)

    response = await async_client.post(
        "/query", json={"question": "What is the net income?", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks_retrieved"] <= 2


@pytest.mark.asyncio
async def test_query_no_documents_returns_not_found_message(async_client):
    """With an empty ChromaDB, the endpoint should still return 200 with a graceful message."""
    response = await async_client.post("/query", json={"question": "What was the EBITDA margin?"})
    assert response.status_code == 200
    body = response.json()
    assert "cannot find" in body["answer"].lower() or body["total_chunks_retrieved"] == 0
