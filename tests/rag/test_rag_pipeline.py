"""
MaintenAI— RAG Knowledge Pipeline Test Suite

Tests document chunking, metadata extraction, SentenceTransformer embeddings,
hybrid retrieval (BM25 + Dense RRF), and Cross-Encoder reranking.
"""

from __future__ import annotations

import pytest

from rag.chunking.markdown_chunker import DocumentChunk, MarkdownManualChunker
from rag.embeddings.embedder import TextEmbedder
from rag.reranking.reranker import CrossEncoderReranker
from rag.retrieval.hybrid import HybridRetriever


@pytest.mark.rag
def test_markdown_chunker_hierarchy_and_codes():
    doc_text = """# Industrial Turbine Maintenance
## 1. Safety Protocols
Always isolate electrical connections before inspection.
## 2. Bearing Inspection
When vibration exceeds 4.5g, check for CODE-BPFO outer race defects.
Trigger emergency shutdown if TWF or HDF conditions appear.
"""
    chunker = MarkdownManualChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(
        doc_text, source_file="turbine_manual.md", equipment_type="turbofan"
    )

    assert len(chunks) >= 2
    assert chunks[0].equipment_type == "turbofan"
    assert chunks[0].source_file == "turbine_manual.md"

    # Verify failure code metadata extraction
    bearing_chunk = [c for c in chunks if "Bearing Inspection" in c.section_title][0]
    codes = bearing_chunk.metadata.get("failure_codes", [])
    assert "HDF" in codes or "TWF" in codes or "CODE-BPFO" in codes


@pytest.mark.rag
def test_bm25_and_hybrid_retrieval():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            content="Turbofan high pressure compressor blades require inspection when RUL drops below 15 cycles.",
            source_file="turbofan_manual.md",
            equipment_type="turbofan",
            section_title="Compressor Blades",
            breadcrumbs=["Turbofan", "Compressor"],
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            content="Bearing lubrication protocol requires NLGI 2 lithium grease application for BPFO vibration.",
            source_file="bearing_manual.md",
            equipment_type="bearing",
            section_title="Lubrication",
            breadcrumbs=["Bearing", "Lubrication"],
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            content="CNC milling machine heat dissipation failure HDF occurs when coolant differential is low.",
            source_file="cnc_manual.md",
            equipment_type="cnc_mill",
            section_title="Coolant System",
            breadcrumbs=["CNC", "Coolant"],
        ),
    ]

    embedder = TextEmbedder()
    retriever = HybridRetriever(embedder=embedder)
    retriever.index(chunks)

    # Query 1: Exact keyword match for HDF
    results_hdf = retriever.retrieve(query="heat dissipation failure HDF", top_k=2)
    assert len(results_hdf) > 0
    assert results_hdf[0][0].equipment_type == "cnc_mill"

    # Query 2: Semantic query for bearing vibration grease
    results_bearing = retriever.retrieve(query="bearing grease and vibration", top_k=2)
    assert len(results_bearing) > 0
    assert results_bearing[0][0].equipment_type == "bearing"

    # Test reranker
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query="bearing grease", candidates=results_bearing, top_k=1)
    assert len(reranked) == 1
