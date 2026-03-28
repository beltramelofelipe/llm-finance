from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitation]
    model: str
    total_chunks_retrieved: int
