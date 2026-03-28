import chromadb
from chromadb.config import Settings as ChromaSettings


def create_chroma_client(persist_dir: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collections(
    client: chromadb.ClientAPI, collection_name: str
) -> tuple[chromadb.Collection, chromadb.Collection]:
    chunks = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    registry = client.get_or_create_collection(
        name=f"{collection_name}_registry",
        metadata={"hnsw:space": "cosine"},
    )
    return chunks, registry
