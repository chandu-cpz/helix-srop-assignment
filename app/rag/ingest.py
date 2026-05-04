"""RAG ingest CLI."""
import argparse
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai.errors import ClientError

from app.rag.embeddings import embed_documents
from app.rag.store import get_docs_collection

FRONTMATTER_PATTERN = re.compile(r"^---\n(?P<frontmatter>.*?)\n---\n?", re.DOTALL)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document: str
    metadata: dict[str, Any]


def _strip_frontmatter(text: str) -> str:
    return FRONTMATTER_PATTERN.sub("", text, count=1).strip()


def _split_large_section(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            split_at = max(
                text.rfind("\n\n", start + chunk_size // 2, end),
                text.rfind("\n", start + chunk_size // 2, end),
                text.rfind(". ", start + chunk_size // 2, end),
            )
            if split_at <= start:
                split_at = end
        else:
            split_at = end

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= len(text):
            break
        start = max(split_at - overlap, start + 1)

    return chunks


def make_chunk_id(file_path: Path, chunk_index: int) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", file_path.stem.lower()).strip("_")
    return f"chunk_{stem}_{chunk_index}"


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split markdown text into heading-aware chunks with overlap."""
    body = _strip_frontmatter(text)
    if not body:
        return []

    sections: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_split_large_section(section, chunk_size, overlap))

    return [chunk for chunk in chunks if chunk.strip()]


def extract_metadata(file_path: Path, text: str) -> dict:
    """Extract frontmatter metadata for one markdown document."""
    metadata = {
        "source": file_path.name,
        "title": file_path.stem.replace("-", " ").title(),
        "product_area": "general",
    }

    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return metadata

    for raw_line in match.group("frontmatter").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip()
        normalized_value = value.strip().strip('"')
        if normalized_key == "tags":
            tags = normalized_value.strip("[]")
            metadata[normalized_key] = ",".join(
                tag.strip() for tag in tags.split(",") if tag.strip()
            )
        else:
            metadata[normalized_key] = normalized_value

    return metadata


def _batch_records(records: list[ChunkRecord], batch_size: int) -> list[list[ChunkRecord]]:
    return [records[start : start + batch_size] for start in range(0, len(records), batch_size)]


def _retry_delay_seconds(exc: ClientError, fallback: float) -> float:
    message = str(exc)
    retry_match = re.search(r"retry(?: in|Delay[^0-9]+)([0-9.]+)s", message, re.IGNORECASE)
    if retry_match is None:
        return fallback
    return max(float(retry_match.group(1)), fallback)


async def _embed_documents_with_quota_retry(
    documents: list[str],
    titles: list[str],
    max_retries: int,
) -> list[list[float]]:
    attempt = 0
    while True:
        try:
            return await embed_documents(documents, titles=titles)
        except ClientError as exc:
            attempt += 1
            if "RESOURCE_EXHAUSTED" not in str(exc) or attempt > max_retries:
                raise
            delay = _retry_delay_seconds(exc, fallback=min(60.0, 15.0 * attempt))
            print(f"  quota exhausted; retrying in {delay:.1f}s")
            await asyncio.sleep(delay)


async def ingest_directory(
    docs_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int = 64,
    batch_delay: float = 20.0,
    max_retries: int = 3,
) -> None:
    """Walk a docs directory, embed all markdown chunks, and upsert them into Chroma."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    md_files = sorted(docs_path.rglob("*.md"))
    collection = get_docs_collection()
    records: list[ChunkRecord] = []
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(text, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")

        for chunk_index, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(file_path, chunk_index),
                    document=chunk,
                    metadata={**metadata, "chunk_index": chunk_index},
                )
            )

    print(f"Embedding and upserting {len(records)} chunks in batches of {batch_size}")
    batches = _batch_records(records, batch_size)
    for batch_number, batch in enumerate(batches, start=1):
        embeddings = await _embed_documents_with_quota_retry(
            [record.document for record in batch],
            titles=[str(record.metadata.get("title", "")) for record in batch],
            max_retries=max_retries,
        )
        collection.upsert(
            ids=[record.chunk_id for record in batch],
            documents=[record.document for record in batch],
            metadatas=[record.metadata for record in batch],
            embeddings=embeddings,
        )
        print(f"  batch {batch_number}: {len(batch)} chunks")
        if batch_delay > 0 and batch_number < len(batches):
            await asyncio.sleep(batch_delay)

    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-delay", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    asyncio.run(
        ingest_directory(
            args.path,
            args.chunk_size,
            args.chunk_overlap,
            batch_size=args.batch_size,
            batch_delay=args.batch_delay,
            max_retries=args.max_retries,
        )
    )


if __name__ == "__main__":
    main()
