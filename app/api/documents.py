import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.dependencies import get_ingestion_service
from app.schemas.documents import DocumentListResponse, DocumentRecord, UploadResponse
from app.services.ingestion import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
):
    """Upload and ingest a PDF financial document."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Also accept octet-stream for clients that don't set content-type correctly
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail="Only PDF files are accepted. Please upload a .pdf file.",
            )

    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {settings.max_upload_size_mb} MB.",
        )

    filename = file.filename or "document.pdf"
    return await ingestion_service.ingest_document(file_bytes, filename)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """List all indexed financial documents."""
    results = ingestion_service._registry.get(include=["metadatas", "documents"])

    documents: list[DocumentRecord] = []
    for meta in results.get("metadatas") or []:
        documents.append(
            DocumentRecord(
                document_id=meta["document_id"],
                filename=meta["filename"],
                page_count=meta["page_count"],
                chunk_count=meta["chunk_count"],
                uploaded_at=datetime.fromisoformat(meta["uploaded_at"]),
                size_bytes=meta["size_bytes"],
            )
        )

    return DocumentListResponse(documents=documents, total=len(documents))
