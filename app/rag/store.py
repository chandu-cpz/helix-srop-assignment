from functools import lru_cache

from chromadb import PersistentClient
from chromadb.api import Collection

from app.settings import settings

COLLECTION_NAME = "helix_docs"


@lru_cache(maxsize=1)
def get_docs_collection() -> Collection:
    client = PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
    )
