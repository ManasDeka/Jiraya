"""
Fallback Node
--------------
Triggered when validator fails after max retries.
Returns a safe, honest fallback message to the user.
"""

from rag.state import RAGState


def fallback_node(state: RAGState) -> RAGState:
    """
    Fallback Node.
    Returns a safe message when answer cannot be validated
    after all retry attempts are exhausted.
    """
    print("[Fallback] ⚠️ Max retries exceeded — returning fallback response")
    return {
        **state,
        "answer": (
            "I was unable to find a confident answer from the available documents. "
            "Please try rephrasing your question or contact your enterprise support team."
        ),
        "validation_result": "FALLBACK",
    }
