import logging

from fastapi import APIRouter, Depends

from app.dependencies import get_generation_service, get_retrieval_service
from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
):
    """Ask a natural language question about the indexed financial documents."""
    chunks = await retrieval_service.retrieve(body.question, top_k=body.top_k)

    if not chunks:
        return QueryResponse(
            question=body.question,
            answer="I cannot find this information in the provided documents.",
            sources=[],
            model=generation_service.model_name,
            total_chunks_retrieved=0,
        )

    answer, sources = await generation_service.generate_answer(body.question, chunks)

    return QueryResponse(
        question=body.question,
        answer=answer,
        sources=sources,
        model=generation_service.model_name,
        total_chunks_retrieved=len(chunks),
    )
