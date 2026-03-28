import asyncio
import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.core.exceptions import DocumentAlreadyExistsError, EmbeddingServiceError, PDFExtractionError
from app.schemas.documents import UploadResponse
from app.utils.pdf import extract_pages

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        chunk_collection: chromadb.Collection,
        registry_collection: chromadb.Collection,
        embeddings: OpenAIEmbeddings,
        settings: Settings,
    ):
        self._chunks = chunk_collection
        self._registry = registry_collection
        self._embeddings = embeddings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def ingest_document(self, file_bytes: bytes, filename: str) -> UploadResponse:
        document_id = hashlib.sha256(file_bytes).hexdigest()[:16]

        # Check for duplicates
        existing = self._registry.get(ids=[document_id])
        if existing["ids"]:
            raise DocumentAlreadyExistsError(document_id)

        # Write to temp file for Docling
        tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp_file.write(file_bytes)
            tmp_file.flush()
            tmp_file.close()

            try:
                pages = await extract_pages(tmp_file.name)
            except Exception as exc:
                logger.exception("Docling extraction failed for '%s'", filename)
                raise PDFExtractionError(str(exc)) from exc
        finally:
            os.unlink(tmp_file.name)

        if not pages:
            raise PDFExtractionError("No text could be extracted from the PDF.")

        page_count = len(pages)
        uploaded_at = datetime.now(timezone.utc).isoformat()
        size_bytes = len(file_bytes)

        # Build chunks across all pages
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        chunk_index = 0

        for page in pages:
            page_chunks = self._splitter.split_text(page["text"])
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue
                chunk_id = f"{document_id}__chunk_{chunk_index:04d}"
                ids.append(chunk_id)
                texts.append(chunk_text)
                metadatas.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "page_count": page_count,
                        "uploaded_at": uploaded_at,
                        "size_bytes": size_bytes,
                    }
                )
                chunk_index += 1

        if not ids:
            raise PDFExtractionError("PDF produced no usable text chunks.")

        chunk_count = len(ids)

        # Update chunk_count in all metadatas now that we know the total
        for meta in metadatas:
            meta["chunk_count"] = chunk_count

        # Generate embeddings
        try:
            loop = asyncio.get_event_loop()
            embedding_vectors = await loop.run_in_executor(
                None, self._embeddings.embed_documents, texts
            )
        except Exception as exc:
            logger.exception("Embedding service error")
            raise EmbeddingServiceError(str(exc)) from exc

        # Store chunks
        self._chunks.add(
            ids=ids,
            embeddings=embedding_vectors,
            documents=texts,
            metadatas=metadatas,
        )

        # Register document (uses a zero vector since registry is never similarity-searched)
        dummy_vector = [0.0] * len(embedding_vectors[0])
        self._registry.add(
            ids=[document_id],
            embeddings=[dummy_vector],
            documents=[filename],
            metadatas=[
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page_count": page_count,
                    "chunk_count": chunk_count,
                    "uploaded_at": uploaded_at,
                    "size_bytes": size_bytes,
                }
            ],
        )

        logger.info("Ingested '%s' → %d chunks, %d pages", filename, chunk_count, page_count)
        return UploadResponse(
            document_id=document_id,
            filename=filename,
            chunk_count=chunk_count,
            message=f"Document '{filename}' successfully ingested with {chunk_count} chunks.",
        )
