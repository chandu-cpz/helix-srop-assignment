"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.
"""
from dataclasses import dataclass

from app.rag.embeddings import embed_query
from app.rag.store import get_docs_collection


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """Search the vector store for the top-k documentation chunks."""
    if k <= 0:
        return []

    query_embedding = await embed_query(query)
    where = {"product_area": product_area} if product_area else None
    result = get_docs_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    chunks: list[DocChunk] = []
    for chunk_id, content, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=False
    ):
        score = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                score=score,
                content=content,
                metadata=dict(metadata or {}),
            )
        )

    return chunks
