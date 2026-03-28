from langchain_openai import OpenAIEmbeddings

from app.config import Settings


def create_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key.get_secret_value(),
    )
