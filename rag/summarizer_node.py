"""
Summarizer Node
----------------
Generates answer from reranked chunks using Azure OpenAI.
Normal mode  → attempt 1 (retry_count == 0)
Strict mode  → attempt 2 (retry_count > 0)

Strict mode adds extra instruction to not infer beyond context.
Includes document citations in the answer.
"""

from openai import AzureOpenAI
from rag.state import RAGState
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    CHAT_DEPLOYMENT_NAME,
    SUMMARIZER_TEMPERATURE,
    SUMMARIZER_MAX_TOKENS,
)

# ── Azure OpenAI Client ──────────────────────────────────────────────
_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


def summarizer_node(state: RAGState) -> RAGState:
    """
    Summarizer Node.
    Builds context from reranked chunks and generates
    a cited answer using Azure OpenAI.
    """

    if state.get("guardrail_triggered"):
        print("[Summarizer] Skipping — guardrail triggered")
        return state

    retry_count = state["retry_count"]
    chunks = state["reranked_chunks"]

    # ── Build Context ────────────────────────────────────────────────
    context_parts = []
    for chunk in chunks:
        doc_name = chunk["metadata"].get("doc_name", "Unknown Document")
        page = chunk["metadata"].get("page_number", "N/A")
        context_parts.append(
            f"[Source: {doc_name} | Page: {page}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    # ── Strict Mode Instruction ──────────────────────────────────────
    strict_instruction = ""
    if retry_count > 0:
        strict_instruction = (
            "\n⚠️ STRICT MODE: Do NOT infer, assume, or go beyond the provided context. "
            "If the answer is not explicitly in the context, say: 'Information not available.'"
        )
        print("[Summarizer] Running in STRICT MODE (retry attempt)")
    else:
        print("[Summarizer] Running in normal mode")

    # ── Prompt ───────────────────────────────────────────────────────
    system_prompt = f"""You are a precise enterprise document assistant.

Answer ONLY using the provided context below.
Do NOT fabricate information.
If the answer is not found in context, respond with: "Information not available in the provided documents."
Always cite the source document at the end of relevant sentences like: (Source: <doc_name>)
{strict_instruction}"""

    user_prompt = f"""Question: {state['cleaned_question']}

Context:
{context_text}

Provide a clear, concise answer with document citations."""

    response = _client.chat.completions.create(
        model=CHAT_DEPLOYMENT_NAME,
        temperature=SUMMARIZER_TEMPERATURE,
        max_tokens=SUMMARIZER_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()
    print(f"[Summarizer] ✅ Answer generated ({len(answer)} chars)")

    return {**state, "answer": answer}
