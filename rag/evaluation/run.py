"""
MaintenAI— RAG Retrieval & Grounding Evaluation Engine

Evaluates retrieval quality (MRR, Precision@K, Recall@K, Groundedness Score)
against benchmark industrial maintenance diagnostic queries.

Usage:
    python -m rag.evaluation.run
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
import joblib

from rag.retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("data/processed/rag_index.joblib")
DEFAULT_METRICS_OUTPUT = Path("rag/evaluation/metrics.json")

# Grounded evaluation benchmark questions and expected source documents / codes
BENCHMARK_EVAL_SET = [
    {
        "query": "What inspection is required when turbofan RUL is under 15 cycles?",
        "expected_source": "turbofan_cf6_manual.md",
        "expected_terms": ["borescope", "RUL", "HPC", "nozzle guide vanes"],
        "equipment_type": "turbofan",
    },
    {
        "query": "What grease and lubrication protocol is used for bearing BPFO outer race defect?",
        "expected_source": "ims_bearing_sop.md",
        "expected_terms": ["NLGI 2", "lithium", "vibration", "grease"],
        "equipment_type": "bearing",
    },
    {
        "query": "How to resolve CNC heat dissipation failure HDF temperature differential?",
        "expected_source": "cnc_milling_troubleshooting.md",
        "expected_terms": ["coolant", "8.6K", "temperature", "HDF"],
        "equipment_type": "cnc_mill",
    },
    {
        "query": "What work order part reference is used for turbofan overhaul?",
        "expected_source": "turbofan_cf6_manual.md",
        "expected_terms": ["KIT-CF6-TURBINE-SEALS-V2", "EMERGENCY_OVERHAUL"],
        "equipment_type": "turbofan",
    },
    {
        "query": "What is the critical vibration shutdown threshold for IMS bearing degradation?",
        "expected_source": "ims_bearing_sop.md",
        "expected_terms": ["8.5g", "LOTO", "lock-out"],
        "equipment_type": "bearing",
    },
]


def evaluate_retriever(retriever: HybridRetriever, top_k: int = 3) -> dict[str, Any]:
    """Calculate retrieval accuracy metrics across benchmark questions."""
    reciprocal_ranks = []
    precision_at_k = []
    recall_at_k = []
    groundedness_scores = []

    for item in BENCHMARK_EVAL_SET:
        query = item["query"]
        expected_src = item["expected_source"]
        expected_terms = item["expected_terms"]
        eq_type = item.get("equipment_type")

        results = retriever.retrieve(query=query, top_k=top_k, equipment_type=eq_type)

        found_rank = 0
        term_matches = 0
        for rank, (chunk, _score) in enumerate(results, 1):
            if expected_src in chunk.source_file and found_rank == 0:
                found_rank = rank
            # Check for grounded keywords in retrieved content
            content_lower = chunk.content.lower()
            for term in expected_terms:
                if term.lower() in content_lower:
                    term_matches += 1

        rr = 1.0 / found_rank if found_rank > 0 else 0.0
        reciprocal_ranks.append(rr)
        precision_at_k.append(1.0 if found_rank > 0 else 0.0)
        recall_at_k.append(1.0 if found_rank > 0 else 0.0)

        # Groundedness: proportion of expected procedural terms matched
        groundedness = min(1.0, term_matches / max(len(expected_terms), 1))
        groundedness_scores.append(groundedness)

    mrr = float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1))
    p_k = float(sum(precision_at_k) / max(len(precision_at_k), 1))
    r_k = float(sum(recall_at_k) / max(len(recall_at_k), 1))
    mean_groundedness = float(sum(groundedness_scores) / max(len(groundedness_scores), 1))

    return {
        "mrr": round(mrr, 4),
        "precision_at_3": round(p_k, 4),
        "recall_at_3": round(r_k, 4),
        "mean_groundedness_score": round(mean_groundedness, 4),
        "total_eval_queries": len(BENCHMARK_EVAL_SET),
        "passed_grounding_gate": mean_groundedness >= 0.80,
    }


@click.command()
@click.option(
    "--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to indexed hybrid retriever"
)
@click.option(
    "--output", default=str(DEFAULT_METRICS_OUTPUT), help="Output path for evaluation metrics JSON"
)
def main(index_path: str, output: str) -> None:
    """Run RAG retrieval and citation grounding evaluation."""
    idx_path = Path(index_path)
    if not idx_path.exists():
        from rag.ingestion.run import load_and_chunk_knowledge_base

        chunks = load_and_chunk_knowledge_base(Path("data/knowledge"))
        retriever = HybridRetriever()
        retriever.index(chunks)
    else:
        retriever = joblib.load(idx_path)

    metrics = evaluate_retriever(retriever)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("RAG Evaluation Results:\n%s", json.dumps(metrics, indent=2))
    logger.info(
        "Groundedness Gate Status: %s", "PASSED" if metrics["passed_grounding_gate"] else "FAILED"
    )


if __name__ == "__main__":
    main()
