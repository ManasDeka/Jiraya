"""
LangGraph RAG Workflow
-----------------------
Full agentic RAG pipeline assembled as a LangGraph StateGraph.

Flow:
  guardrail → classifier → retriever → reranker
  → summarizer → validator → output_guardrail
  → END (VALID) / retriever (retry) / fallback (max retries)
"""

from langgraph.graph import StateGraph, END

from rag.state import RAGState
from rag.guardrail_node import guardrail_node
from rag.classifier_node import classifier_node
from rag.retriever_node import retriever_node
from rag.reranker_node import reranker_node
from rag.summarizer_node import summarizer_node
from rag.validator_node import validator_node
from rag.output_guardrail_node import output_guardrail_node
from rag.fallback_node import fallback_node
from config import MAX_RETRY_COUNT


# ── Conditional Router ───────────────────────────────────────────────
def validation_router(state: RAGState) -> str:
    """
    Routes after validator based on result and retry count.

    VALID        → output_guardrail → END
    INVALID + retry available → back to retriever with incremented count
    INVALID + max retries hit → fallback
    """

    # If input guardrail blocked the query → go straight to END
    if state.get("guardrail_triggered"):
        print("[Router] Guardrail triggered — routing to END")
        return "end"

    validation_result = state.get("validation_result", "INVALID")
    retry_count = state.get("retry_count", 0)

    if validation_result == "VALID":
        print("[Router] VALID — routing to output guardrail")
        return "output_guardrail"

    if retry_count < MAX_RETRY_COUNT:
        print(f"[Router] INVALID — retry {retry_count + 1} — routing back to retriever")
        # Increment retry count in state
        state["retry_count"] += 1
        return "retriever"

    print("[Router] INVALID — max retries hit — routing to fallback")
    return "fallback"


# ── Guardrail Router ─────────────────────────────────────────────────
def guardrail_router(state: RAGState) -> str:
    """
    Routes after input guardrail.
    If triggered → skip to END immediately with safe message.
    If clean     → proceed to classifier.
    """
    if state.get("guardrail_triggered"):
        print("[Router] Input blocked by guardrail — routing to END")
        return "end"
    return "classifier"


# ── Build Graph ──────────────────────────────────────────────────────
def build_rag_graph():
    """
    Builds and compiles the full LangGraph RAG workflow.
    Returns compiled app ready for invocation.
    """
    workflow = StateGraph(RAGState)

    # ── Register Nodes ───────────────────────────────────────────────
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("fallback", fallback_node)

    # ── Entry Point ──────────────────────────────────────────────────
    workflow.set_entry_point("guardrail")

    # ── Conditional Edge After Input Guardrail ───────────────────────
    workflow.add_conditional_edges(
        "guardrail",
        guardrail_router,
        {
            "classifier": "classifier",
            "end": END,
        },
    )

    # ── Linear Edges ─────────────────────────────────────────────────
    workflow.add_edge("classifier", "retriever")
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "summarizer")
    workflow.add_edge("summarizer", "validator")

    # ── Conditional Edge After Validator ─────────────────────────────
    workflow.add_conditional_edges(
        "validator",
        validation_router,
        {
            "output_guardrail": "output_guardrail",
            "retriever": "retriever",
            "fallback": "fallback",
            "end": END,
        },
    )

    # ── Terminal Edges ────────────────────────────────────────────────
    workflow.add_edge("output_guardrail", END)
    workflow.add_edge("fallback", END)

    app = workflow.compile()
    print("[Graph] ✅ RAG graph compiled successfully")
    return app
