"""
Reranker Node
--------------
Uses a cross-encoder model to rerank retrieved chunks
by relevance to the user query.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (runs locally)
Toggle: ENABLE_RERANKER in config.py

If disabled → passes retrieved_chunks directly as reranked_chunks.
If enabled  → scores each chunk against query, keeps top RERANKER_TOP_N.
"""

from rag.state import RAGState
from config import ENABLE_RERANKER, RERANKER_MODEL, RERANKER_TOP_N


def reranker_node(state: RAGState) -> RAGState:
    """
    Reranker Node.
    Reranks retrieved chunks using cross-encoder scoring.

    Cross-encoder takes (query, chunk) pair and outputs
    a relevance score — more accurate than embedding similarity alone.
    """

    if state.get("guardrail_triggered"):
        print("[Reranker] Skipping — guardrail triggered")
        return state

    chunks = state["retrieved_chunks"]

    # ── Toggle Check ─────────────────────────────────────────────────
    if not ENABLE_RERANKER:
        print("[Reranker] Reranker disabled — passing chunks through")
        return {**state, "reranked_chunks": chunks}

    if not chunks:
        print("[Reranker] No chunks to rerank")
        return {**state, "reranked_chunks": chunks}

    # ── Lazy import to avoid loading model unless needed ─────────────
    from sentence_transformers import CrossEncoder

    query = state["cleaned_question"]
    print(f"[Reranker] Reranking {len(chunks)} chunks with '{RERANKER_MODEL}'...")

    # Load cross-encoder model
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    # Create (query, chunk_text) pairs for scoring
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs
    scores = cross_encoder.predict(pairs)

    # Attach scores to chunks
    scored_chunks = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]

    # Sort by score descending and keep top N
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_chunks = scored_chunks[:RERANKER_TOP_N]

    print(f"[Reranker] ✅ Kept top {len(top_chunks)} chunks after reranking")
    for i, chunk in enumerate(top_chunks):
        print(f"   [{i+1}] Score: {chunk['rerank_score']:.4f} | "
              f"Doc: {chunk['metadata'].get('doc_name', 'unknown')}")

    return {**state, "reranked_chunks": top_chunks}
