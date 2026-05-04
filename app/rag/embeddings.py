from functools import lru_cache

from google import genai
from google.genai import types

from app.settings import settings


def _prepare_query(query: str) -> str:
    return f"task: search result | query: {query.strip()}"


def _prepare_document(content: str, title: str | None) -> str:
    normalized_title = title.strip() if title else "none"
    return f"task: search result | title: {normalized_title} | text: {content.strip()}"


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    if settings.google_api_key:
        return genai.Client(api_key=settings.google_api_key)
    return genai.Client()


async def embed_query(query: str) -> list[float]:
    response = await get_genai_client().aio.models.embed_content(
        model=settings.embedding_model,
        contents=_prepare_query(query),
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
    )
    return list(response.embeddings[0].values)


async def embed_documents(
    documents: list[str],
    titles: list[str | None] | None = None,
) -> list[list[float]]:
    if not documents:
        return []

    normalized_titles = titles or [None] * len(documents)
    contents = [
        types.Content(parts=[types.Part.from_text(text=_prepare_document(document, title))])
        for document, title in zip(documents, normalized_titles, strict=True)
    ]
    response = await get_genai_client().aio.models.embed_content(
        model=settings.embedding_model,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
    )
    return [list(embedding.values) for embedding in response.embeddings]
