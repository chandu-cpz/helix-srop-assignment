"""Focused unit tests for RAG chunking and retrieval."""
import pytest
from chromadb import PersistentClient

from app.agents.tools.search_docs import search_docs
from app.rag.ingest import chunk_markdown


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(monkeypatch, tmp_path):
    """search_docs must return chunk IDs and scores in [0, 1]."""
    collection = PersistentClient(path=tmp_path / "chroma").get_or_create_collection(
        name="helix_docs"
    )
    collection.upsert(
        ids=["chunk_deploy_keys_0", "chunk_builds_0"],
        documents=[
            "Rotate a deploy key by adding a new key and removing the old one.",
            "Build status reference.",
        ],
        metadatas=[
            {"product_area": "security", "source": "deploy-keys.md", "title": "Deploy Keys"},
            {"product_area": "builds", "source": "builds.md", "title": "Builds"},
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    async def fake_embed_query(query: str) -> list[float]:
        assert "deploy key" in query.lower()
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr("app.agents.tools.search_docs.embed_query", fake_embed_query)
    monkeypatch.setattr("app.agents.tools.search_docs.get_docs_collection", lambda: collection)

    results = await search_docs("how to rotate a deploy key", k=3)

    assert len(results) > 0
    assert all(result.chunk_id for result in results)
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert results[0].chunk_id == "chunk_deploy_keys_0"


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    chunks = chunk_markdown(text, chunk_size=40, overlap=10)

    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)
