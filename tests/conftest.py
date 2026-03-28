"""
Shared fixtures for the test suite.

Strategy:
- ChromaDB uses EphemeralClient (in-memory, no disk I/O)
- OpenAI embeddings are mocked to return deterministic zero vectors
- ChatOpenAI is mocked to return a fixed answer string
- A minimal valid single-page PDF is generated inline (no external files)
"""

import struct
import zlib
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.core.chroma import get_collections
from app.core.exceptions import register_exception_handlers
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService

# Dimensionality for text-embedding-3-small
EMBEDDING_DIM = 1536
FIXED_ANSWER = "Net revenue was R$ 10 billion in Q3 2024 [report.pdf | page 1]."


# ---------------------------------------------------------------------------
# Minimal valid PDF (no external dependency)
# ---------------------------------------------------------------------------

def make_minimal_pdf(text: str = "Net revenue: R$ 10 billion. Operating income: R$ 2 billion.") -> bytes:
    """Build a minimal but valid single-page PDF with the given text."""
    # We construct a very small PDF manually.
    objects: list[bytes] = []

    def obj(n: int, content: bytes) -> bytes:
        return f"{n} 0 obj\n".encode() + content + b"\nendobj\n"

    # Object 1: Catalog
    objects.append(obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"))
    # Object 2: Pages
    objects.append(obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    # Object 4: Font
    objects.append(obj(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    # Object 5: Content stream
    content_stream = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode()
    objects.append(
        obj(
            5,
            f"<< /Length {len(content_stream)} >>\nstream\n".encode()
            + content_stream
            + b"\nendstream",
        )
    )

    # Object 3: Page (references content and font)
    objects.append(
        obj(
            3,
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Contents 5 0 R "
            b"/Resources << /Font << /F1 4 0 R >> >> >>",
        )
    )

    header = b"%PDF-1.4\n"
    body = b"".join(objects)
    xref_offset = len(header) + len(body)

    xref = b"xref\n0 6\n0000000000 65535 f \n"
    # Simplified: just mark all as free (enough to be parseable by most readers)
    xref += b"0000000000 00000 n \n" * 5

    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n".encode()
        + b"%%EOF"
    )

    return header + body + xref + trailer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    return make_minimal_pdf()


@pytest.fixture
def mock_embeddings():
    """Returns an OpenAIEmbeddings mock that yields deterministic zero vectors."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [[0.0] * EMBEDDING_DIM for _ in texts]
    mock.embed_query.return_value = [0.0] * EMBEDDING_DIM
    return mock


@pytest.fixture
def chroma_collections():
    """In-memory ChromaDB client + collections, reset per test."""
    client = chromadb.EphemeralClient()
    chunks, registry = get_collections(client, "test_documents")
    return chunks, registry


@pytest.fixture
def ingestion_service(chroma_collections, mock_embeddings):
    chunks, registry = chroma_collections
    settings = Settings(
        openai_api_key="sk-test",
        chroma_persist_dir="/tmp/test_chroma",
        chroma_collection_name="test_documents",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_top_k=3,
    )
    return IngestionService(chunks, registry, mock_embeddings, settings)


@pytest.fixture
def retrieval_service(chroma_collections, mock_embeddings):
    chunks, _ = chroma_collections
    settings = Settings(
        openai_api_key="sk-test",
        chroma_persist_dir="/tmp/test_chroma",
        chroma_collection_name="test_documents",
        retrieval_top_k=3,
    )
    return RetrievalService(chunks, mock_embeddings, settings)


@pytest.fixture
def generation_service():
    settings = Settings(
        openai_api_key="sk-test",
        openai_chat_model="gpt-4o-mini",
    )
    service = GenerationService(settings)
    # Patch the LangChain chain's ainvoke
    mock_chain = AsyncMock(return_value=FIXED_ANSWER)
    service._chain = mock_chain
    return service


@pytest.fixture
def test_app(chroma_collections, mock_embeddings, generation_service):
    """Full FastAPI app wired with test doubles."""
    from app.api.documents import router as doc_router
    from app.api.query import router as query_router

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(doc_router, prefix="/documents", tags=["Documents"])
    app.include_router(query_router, tags=["Query"])

    chunks, registry = chroma_collections
    settings = Settings(
        openai_api_key="sk-test",
        chroma_persist_dir="/tmp/test_chroma",
        chroma_collection_name="test_documents",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_top_k=3,
        max_upload_size_mb=10,
    )

    app.state.ingestion_service = IngestionService(chunks, registry, mock_embeddings, settings)
    app.state.retrieval_service = RetrievalService(chunks, mock_embeddings, settings)
    app.state.generation_service = generation_service
    return app


@pytest.fixture
async def async_client(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        yield client
