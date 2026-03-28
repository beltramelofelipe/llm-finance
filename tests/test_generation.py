"""Unit tests for GenerationService."""

import pytest

from tests.conftest import FIXED_ANSWER


def _make_chunks(n: int = 2) -> list[dict]:
    return [
        {
            "text": f"Revenue in Q{i} was R$ {i * 5} billion.",
            "metadata": {
                "document_id": "abc123",
                "filename": "report.pdf",
                "page_number": i,
                "chunk_index": i - 1,
            },
            "relevance_score": 0.9 - i * 0.1,
        }
        for i in range(1, n + 1)
    ]


@pytest.mark.asyncio
async def test_generate_answer_returns_string(generation_service):
    chunks = _make_chunks(2)
    answer, sources = await generation_service.generate_answer("What was Q1 revenue?", chunks)
    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_generate_answer_matches_mock(generation_service):
    chunks = _make_chunks(2)
    answer, _ = await generation_service.generate_answer("What was Q1 revenue?", chunks)
    assert answer == FIXED_ANSWER


@pytest.mark.asyncio
async def test_sources_built_from_chunks(generation_service):
    chunks = _make_chunks(3)
    _, sources = await generation_service.generate_answer("What was revenue?", chunks)

    assert len(sources) == 3
    for i, source in enumerate(sources):
        assert source.document_id == "abc123"
        assert source.filename == "report.pdf"
        assert source.page_number == i + 1
        assert source.chunk_index == i
        assert 0.0 <= source.relevance_score <= 1.0
        assert len(source.excerpt) <= 300


@pytest.mark.asyncio
async def test_sources_excerpt_truncated(generation_service):
    long_text = "A" * 500
    chunks = [
        {
            "text": long_text,
            "metadata": {
                "document_id": "x",
                "filename": "f.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
            "relevance_score": 0.8,
        }
    ]
    _, sources = await generation_service.generate_answer("question?", chunks)
    assert len(sources[0].excerpt) == 300
