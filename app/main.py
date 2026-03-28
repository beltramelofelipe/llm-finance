import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.core.chroma import create_chroma_client, get_collections
from app.core.embeddings import create_embeddings
from app.core.exceptions import register_exception_handlers
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.getLogger().setLevel(settings.log_level.upper())

    logger.info("Initialising ChromaDB at %s", settings.chroma_persist_dir)
    chroma_client = create_chroma_client(settings.chroma_persist_dir)
    chunk_col, registry_col = get_collections(chroma_client, settings.chroma_collection_name)

    logger.info("Initialising embeddings model: %s", settings.openai_embedding_model)
    embeddings = create_embeddings(settings)

    app.state.ingestion_service = IngestionService(chunk_col, registry_col, embeddings, settings)
    app.state.retrieval_service = RetrievalService(chunk_col, embeddings, settings)
    app.state.generation_service = GenerationService(settings)

    logger.info("Application started successfully.")
    yield
    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    from app.api.documents import router as documents_router
    from app.api.query import router as query_router

    app = FastAPI(
        title="RAG Financial Documents API",
        description=(
            "API para consultas em linguagem natural sobre documentos financeiros (PDFs). "
            "Faça upload de relatórios, balanços e demonstrativos; depois pergunte sobre eles."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(documents_router, prefix="/documents", tags=["Documents"])
    app.include_router(query_router, tags=["Query"])

    return app


app = create_app()
