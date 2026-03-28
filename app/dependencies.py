from fastapi import Request

from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generation_service
