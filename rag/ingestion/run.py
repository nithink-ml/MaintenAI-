"""
MaintenAI— RAG Knowledge Ingestion Pipeline

Parses maintenance manuals and SOPs, creates grounded chunks, and indexes them
into the Hybrid Retriever and Qdrant vector database.

Usage:
    python -m rag.ingestion.run
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import joblib

from rag.chunking.markdown_chunker import DocumentChunk, MarkdownManualChunker
from rag.embeddings.embedder import TextEmbedder
from rag.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_DIR = Path("data/knowledge")
DEFAULT_INDEX_PATH = Path("data/processed/rag_index.joblib")


def load_and_chunk_knowledge_base(knowledge_dir: Path) -> list[DocumentChunk]:
    """Parse all markdown manuals in knowledge directory."""
    chunker = MarkdownManualChunker(chunk_size=512, chunk_overlap=64)
    all_chunks: list[DocumentChunk] = []

    if not knowledge_dir.exists():
        logger.warning("Knowledge directory not found: %s", knowledge_dir)
        return []

    for file_path in sorted(knowledge_dir.glob("*.md")):
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        equipment_type = (
            "turbofan"
            if "turbofan" in file_path.name
            else "bearing"
            if "bearing" in file_path.name
            else "cnc_mill"
            if "cnc" in file_path.name
            else "general"
        )

        chunks = chunker.chunk_document(
            content=content,
            source_file=file_path.name,
            equipment_type=equipment_type,
        )
        all_chunks.extend(chunks)
        logger.info("Chunked %s into %d grounded chunks", file_path.name, len(chunks))

    return all_chunks


@click.command()
@click.option(
    "--knowledge-dir", default=str(DEFAULT_KNOWLEDGE_DIR), help="Directory of markdown manuals"
)
@click.option(
    "--output-index", default=str(DEFAULT_INDEX_PATH), help="Path to save indexed hybrid retriever"
)
def main(knowledge_dir: str, output_index: str) -> None:
    """Ingest maintenance manuals and build hybrid index."""
    k_dir = Path(knowledge_dir)
    chunks = load_and_chunk_knowledge_base(k_dir)

    if not chunks:
        logger.warning("No document chunks created. Please add markdown manuals to %s", k_dir)
        return

    embedder = TextEmbedder()
    retriever = HybridRetriever(embedder=embedder)
    retriever.index(chunks)

    # Save index artifact
    out_path = Path(output_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(retriever, out_path)
    logger.info("Saved RAG Hybrid Index to %s with %d total chunks", out_path, len(chunks))


if __name__ == "__main__":
    main()
