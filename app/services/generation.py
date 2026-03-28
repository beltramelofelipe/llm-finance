import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.schemas.query import SourceCitation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial analyst assistant. Answer the user's question using ONLY the \
provided document excerpts below. Be precise, factual, and concise.

When referencing specific data, cite inline using the format [filename | page N].

If the information needed to answer the question is NOT present in the excerpts, \
respond exactly with: "I cannot find this information in the provided documents."\
"""

HUMAN_PROMPT = """\
Question: {question}

Document Excerpts:
{context}
"""


class GenerationService:
    def __init__(self, settings: Settings):
        self._model_name = settings.openai_chat_model
        self._llm = ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key.get_secret_value(),
            temperature=0,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    async def generate_answer(
        self, question: str, retrieved_chunks: list[dict]
    ) -> tuple[str, list[SourceCitation]]:
        context_parts = []
        for chunk in retrieved_chunks:
            meta = chunk["metadata"]
            header = f"[{meta['filename']} | page {meta['page_number']} | chunk {meta['chunk_index']}]"
            context_parts.append(f"{header}\n{chunk['text']}\n---")
        context_str = "\n\n".join(context_parts)

        answer = await self._chain.ainvoke({"question": question, "context": context_str})

        sources = [
            SourceCitation(
                document_id=chunk["metadata"]["document_id"],
                filename=chunk["metadata"]["filename"],
                page_number=chunk["metadata"]["page_number"],
                chunk_index=chunk["metadata"]["chunk_index"],
                relevance_score=chunk["relevance_score"],
                excerpt=chunk["text"][:300],
            )
            for chunk in retrieved_chunks
        ]

        logger.debug("Generated answer for: %s", question[:80])
        return answer, sources

    @property
    def model_name(self) -> str:
        return self._model_name
