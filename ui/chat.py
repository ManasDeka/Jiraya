"""
Chat Handler
-------------
Handles RAG pipeline execution and returns clean response to UI.
All internal pipeline logs are suppressed — UI only sees final answer.
Supports session tracking for Langfuse observability.
"""

import sys
import os
import time
import re
import uuid


def run_rag_pipeline(question: str, rag_app, session_id: str = None) -> dict:
    """
    Runs the full LangGraph RAG pipeline silently.

    Args:
        question   : Raw user question from UI
        rag_app    : Compiled LangGraph app (built once, reused)
        session_id : Optional session ID for Langfuse tracing

    Returns:
        dict with keys:
            answer            : Final answer string
            domain            : Classified domain
            citations         : List of source citations
            guardrail_triggered: True if PII/profanity blocked query
            validation        : VALID/INVALID/FALLBACK
    """
    from rag.state import RAGState

    # ── Initialize fresh state for every query ───────────────────────
    initial_state: RAGState = {
        "question":           question,
        "cleaned_question":   "",
        "domain":             "",
        "retrieved_chunks":   [],
        "reranked_chunks":    [],
        "answer":             "",
        "validation_result":  "",
        "retry_count":        0,
        "guardrail_triggered": False,
        "output_flagged":     False,
    }

    # ── Suppress all pipeline stdout — keep UI clean ─────────────────
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8")

    try:
        final_state = rag_app.invoke(
            initial_state,
            config={
                "run_id":   session_id or str(uuid.uuid4()),
                "metadata": {"session_id": session_id},
            },
        )
        sys.stdout = old_stdout

    except Exception as e:
        # ── Restore stdout before returning error ────────────────────
        sys.stdout = old_stdout
        print(f"[Chat] ❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "answer":             "An error occurred while processing your question. Please try again.",
            "domain":             "N/A",
            "citations":          [],
            "guardrail_triggered": False,
            "validation":         "ERROR",
            "error":              str(e),
        }

    # ── Handle guardrail-blocked queries ─────────────────────────────
    # Guardrail sets answer directly and stops the pipeline
    if final_state.get("guardrail_triggered", False):
        print(f"[Chat] Query blocked by input guardrail")
        return {
            "answer":             final_state.get("answer", ""),
            "domain":             "N/A",
            "citations":          [],     # No citations for blocked queries
            "guardrail_triggered": True,
            "validation":         "BLOCKED",
        }

    # ── Normal RAG response ──────────────────────────────────────────
    answer    = final_state.get("answer", "No answer generated.")
    domain    = final_state.get("domain", "N/A")
    citations = _extract_citations(answer)

    print(f"[Chat] ✅ Response ready | Domain: {domain} | Citations: {len(citations)}")

    return {
        "answer":             answer,
        "domain":             domain,
        "citations":          citations,
        "guardrail_triggered": False,
        "validation":         final_state.get("validation_result", "N/A"),
    }


def _extract_citations(answer: str) -> list:
    """
    Extracts source citations from the answer text.

    Looks for patterns:
        (Source: filename.pdf)
        (Document: filename.pdf)
        Source: filename.pdf

    Args:
        answer : Raw answer string from summarizer

    Returns:
        List of unique citation strings
    """
    patterns = [
        r'\(Source:\s*([^)]+)\)',
        r'\(Document:\s*([^)]+)\)',
        r'Source:\s*([^\n]+)',
    ]

    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        citations.extend([m.strip() for m in matches])

    # ── Deduplicate while preserving order ───────────────────────────
    seen             = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    return unique_citations


def stream_answer(answer: str):
    """
    Generator that yields answer word by word for streaming effect.
    Used with st.write_stream() in Streamlit UI.

    Args:
        answer : Full answer string

    Yields:
        One word at a time with small delay for streaming feel
    """
    words = answer.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.025)
