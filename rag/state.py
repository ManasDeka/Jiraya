"""
RAG State Definition
---------------------
Shared TypedDict state passed between all LangGraph nodes.
"""

from typing import TypedDict, List


class RAGState(TypedDict):
    question:            str           # Raw user query
    cleaned_question:    str           # After guardrail processing
    domain:              str           # HR / IT / Finance / Operations
    retrieved_chunks:    List[dict]    # Raw chunks from ChromaDB
    reranked_chunks:     List[dict]    # Top chunks after reranking
    answer:              str           # Final generated answer
    validation_result:   str           # VALID / INVALID / FALLBACK
    retry_count:         int           # 0 or 1
    guardrail_triggered: bool          # True if input blocked query
    output_flagged:      bool          # True if output was flagged

# """
# RAG State Definition
# ---------------------
# Shared state passed between all LangGraph nodes.
# Every node receives this state and returns an updated version.
# """

# from typing import TypedDict, List


# class RAGState(TypedDict):
#     question: str               # Raw user query
#     cleaned_question: str       # After PII masking + profanity check
#     domain: str                 # Classified domain: HR/IT/Finance/Operations
#     retrieved_chunks: List[dict] # Chunks from ChromaDB
#     reranked_chunks: List[dict]  # Chunks after cross-encoder reranking
#     answer: str                 # Generated answer from summarizer
#     validation_result: str      # VALID or INVALID from validator
#     retry_count: int            # Tracks retry attempts (0 or 1)
#     guardrail_triggered: bool   # True if input guardrail blocked the query
#     output_flagged: bool        # True if output guardrail flagged the answer
