# FactoryPulse: RAG Knowledge & Retrieval Specification

## 1. RAG Architecture Principles

MaintenAIprevents LLM hallucinations of industrial repair procedures by strictly grounding all diagnostic advice in verified equipment manuals and SOPs.

```
Manual Documents (Markdown) 
  → Hierarchical Header Chunker 
  → Dense Embeddings (SentenceTransformers) + Sparse Index (BM25)
  → Hybrid Retrieval (Reciprocal Rank Fusion)
  → Cross-Encoder Reranker
  → Grounded Citation Injection
```

## 2. Ingestion & Chunking
- **Hierarchical Markdown Chunker**: Chunks text by heading levels (`H1`, `H2`, `H3`), extracting section breadcrumbs and tagging domain failure codes (`TWF`, `HDF`, `PWF`, `CODE-BPFO`).
- **Dense Embedding**: Encodes chunks into 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2`.

## 3. Hybrid Retrieval & Reranking
- **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \frac{0.6}{60 + \text{rank}_{\text{dense}}(d)} + \frac{0.4}{60 + \text{rank}_{\text{sparse}}(d)}$$
- **Cross-Encoder Reranking**: Re-evaluates top 10 candidate chunks using deep cross-attention for high precision.
