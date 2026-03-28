"""Integration tests for document endpoints."""

import io
from unittest.mock import patch, AsyncMock

import pytest


@pytest.mark.asyncio
async def test_upload_pdf_201(async_client, minimal_pdf_bytes):
    with patch(
        "app.services.ingestion.extract_pages",
        return_value=[{"page_number": 1, "text": "Net revenue: R$ 10 billion."}],
    ):
        response = await async_client.post(
            "/documents/upload",
            files={"file": ("report.pdf", io.BytesIO(minimal_pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "report.pdf"
    assert body["chunk_count"] >= 1
    assert "document_id" in body
    assert "message" in body


@pytest.mark.asyncio
async def test_upload_non_pdf_422(async_client):
    response = await async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"plain text content"), "text/plain")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_duplicate_409(async_client, minimal_pdf_bytes):
    with patch(
        "app.services.ingestion.extract_pages",
        return_value=[{"page_number": 1, "text": "Revenue data."}],
    ):
        r1 = await async_client.post(
            "/documents/upload",
            files={"file": ("report.pdf", io.BytesIO(minimal_pdf_bytes), "application/pdf")},
        )
        assert r1.status_code == 201

        r2 = await async_client.post(
            "/documents/upload",
            files={"file": ("report.pdf", io.BytesIO(minimal_pdf_bytes), "application/pdf")},
        )
        assert r2.status_code == 409


@pytest.mark.asyncio
async def test_list_documents_empty(async_client):
    response = await async_client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["documents"] == []


@pytest.mark.asyncio
async def test_list_documents_after_upload(async_client, minimal_pdf_bytes):
    with patch(
        "app.services.ingestion.extract_pages",
        return_value=[{"page_number": 1, "text": "Revenue: R$ 5 billion."}],
    ):
        await async_client.post(
            "/documents/upload",
            files={"file": ("annual.pdf", io.BytesIO(minimal_pdf_bytes), "application/pdf")},
        )

    response = await async_client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    doc = body["documents"][0]
    assert doc["filename"] == "annual.pdf"
    assert doc["chunk_count"] >= 1
    assert doc["page_count"] >= 1
    assert "uploaded_at" in doc
    assert "document_id" in doc
