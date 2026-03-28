from datetime import datetime

from pydantic import BaseModel


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime
    size_bytes: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int
