"""Unit tests for IngestionService."""

import pytest

from app.core.exceptions import DocumentAlreadyExistsError, PDFExtractionError
from tests.conftest import make_minimal_pdf


@pytest.mark.asyncio
async def test_ingest_new_document(ingestion_service, minimal_pdf_bytes):
    """A valid PDF should be ingested and return UploadResponse."""
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.services.ingestion.extract_pages",
        return_value=[{"page_number": 1, "text": "Net revenue: R$ 10 billion. EBITDA: R$ 3 billion."}],
    ):
        result = await ingestion_service.ingest_document(minimal_pdf_bytes, "report.pdf")

    assert result.filename == "report.pdf"
    assert result.chunk_count >= 1
    assert result.document_id  # non-empty


@pytest.mark.asyncio
async def test_ingest_duplicate_raises(ingestion_service, minimal_pdf_bytes):
    """Ingesting the same file twice should raise DocumentAlreadyExistsError."""
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.services.ingestion.extract_pages",
        return_value=[{"page_number": 1, "text": "Revenue: R$ 5 billion."}],
    ):
        await ingestion_service.ingest_document(minimal_pdf_bytes, "report.pdf")
        with pytest.raises(DocumentAlreadyExistsError):
            await ingestion_service.ingest_document(minimal_pdf_bytes, "report.pdf")


@pytest.mark.asyncio
async def test_chunk_metadata_has_page_number(ingestion_service, minimal_pdf_bytes):
    """All stored chunks must carry a valid page_number >= 1."""
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.services.ingestion.extract_pages",
        return_value=[
            {"page_number": 1, "text": "Revenue: R$ 5 billion."},
            {"page_number": 2, "text": "Net income: R$ 1 billion."},
        ],
    ):
        result = await ingestion_service.ingest_document(minimal_pdf_bytes, "annual.pdf")

    results = ingestion_service._chunks.get(include=["metadatas"])
    for meta in results["metadatas"]:
        assert meta["page_number"] >= 1


@pytest.mark.asyncio
async def test_ingest_empty_extraction_raises(ingestion_service, minimal_pdf_bytes):
    """If Docling returns no pages, PDFExtractionError should be raised."""
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "app.services.ingestion.extract_pages",
        return_value=[],
    ):
        with pytest.raises(PDFExtractionError):
            await ingestion_service.ingest_document(minimal_pdf_bytes, "empty.pdf")
