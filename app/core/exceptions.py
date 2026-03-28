import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class DocumentAlreadyExistsError(Exception):
    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' already exists.")


class DocumentNotFoundError(Exception):
    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' not found.")


class PDFExtractionError(Exception):
    pass


class EmbeddingServiceError(Exception):
    pass


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentAlreadyExistsError)
    async def document_already_exists_handler(request: Request, exc: DocumentAlreadyExistsError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(PDFExtractionError)
    async def pdf_extraction_handler(request: Request, exc: PDFExtractionError):
        return JSONResponse(status_code=422, content={"detail": f"PDF extraction failed: {exc}"})

    @app.exception_handler(EmbeddingServiceError)
    async def embedding_service_handler(request: Request, exc: EmbeddingServiceError):
        return JSONResponse(status_code=502, content={"detail": "Embedding service unavailable."})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
