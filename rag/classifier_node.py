"""
Classifier Node
----------------
Reuses the same classify_domain() function from ingestion.
No duplication — same function works for both:
  - Document classification during ingestion
  - Query routing during RAG serving
"""

from rag.state import RAGState
from ingestion.domain_classifier import classify_domain


def classifier_node(state: RAGState) -> RAGState:
    """
    Classifies the cleaned user query into one of:
    HR, IT, Finance, Operations

    Reuses ingestion/domain_classifier.py directly.
    Routes retrieval to the correct ChromaDB collection.
    """

    # If guardrail already blocked query → skip classifier
    if state.get("guardrail_triggered"):
        print("[Classifier] Skipping — guardrail already triggered")
        return state

    cleaned_question = state["cleaned_question"]
    print(f"[Classifier] Classifying query: '{cleaned_question[:80]}...'")

    domain = classify_domain(cleaned_question)
    print(f"[Classifier] ✅ Domain: {domain}")

    return {**state, "domain": domain}
