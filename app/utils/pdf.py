import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_pages_sync(pdf_path: str) -> list[dict]:
    """Extracts text per page from a PDF using Docling. Runs synchronously."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    pages: list[dict] = []
    for page_no, page in enumerate(doc.pages, start=1):
        text_parts = []
        for item in page.cells if hasattr(page, "cells") else []:
            if hasattr(item, "text") and item.text:
                text_parts.append(item.text)

        # Fallback: export full document text if cells are empty
        if not text_parts:
            full_text = doc.export_to_markdown()
            # Return as a single page when granular page data is unavailable
            if full_text.strip():
                return [{"page_number": 1, "text": full_text.strip()}]
            return []

        page_text = "\n".join(text_parts).strip()
        if page_text:
            pages.append({"page_number": page_no, "text": page_text})

    return pages


async def extract_pages(pdf_path: str) -> list[dict]:
    """Async wrapper: runs Docling extraction in a threadpool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_pages_sync, pdf_path)


def count_pages(pdf_path: str) -> int:
    """Returns the number of pages in a PDF using Docling."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return len(list(result.document.pages))
