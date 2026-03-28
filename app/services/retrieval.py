import asyncio
import logging

import chromadb
from langchain_openai import OpenAIEmbeddings

from app.config import Settings

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        chunk_collection: chromadb.Collection,
        embeddings: OpenAIEmbeddings,
        settings: Settings,
    ):
        self._chunks = chunk_collection
        self._embeddings = embeddings
        self._default_top_k = settings.retrieval_top_k

    async def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        k = top_k if top_k is not None else self._default_top_k

        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(None, self._embeddings.embed_query, question)

        results = self._chunks.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": text,
                    "metadata": meta,
                    "relevance_score": round(1.0 - distance, 4),
                }
            )

        logger.debug("Retrieved %d chunks for question: %s", len(chunks), question[:80])
        return chunks
