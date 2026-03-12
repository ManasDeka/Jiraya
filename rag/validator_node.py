"""
Validator Node
---------------
Checks if the generated answer is actually supported
by the retrieved context.

Returns:
  VALID   → answer is grounded in context
  INVALID → answer contains unsupported claims
"""

from openai import AzureOpenAI
from rag.state import RAGState
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    CHAT_DEPLOYMENT_NAME,
    VALIDATOR_TEMPERATURE,
    VALIDATOR_MAX_TOKENS,
)

# ── Azure OpenAI Client ──────────────────────────────────────────────
_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


def validator_node(state: RAGState) -> RAGState:
    """
    Validator Node.
    Verifies the generated answer against the retrieved context.
    Detects hallucinations or unsupported claims.
    """

    if state.get("guardrail_triggered"):
        print("[Validator] Skipping — guardrail triggered")
        return state

    chunks = state["reranked_chunks"]
    context_text = "\n\n".join([chunk["text"] for chunk in chunks])

    system_prompt = """You are a strict answer validator for an enterprise RAG system.

Your job:
- Compare the Answer to the Context provided
- If the Answer is fully supported by the Context → return exactly: VALID
- If the Answer contains ANY information not found in Context → return exactly: INVALID
- Return ONLY one word: VALID or INVALID. Nothing else."""

    user_prompt = f"""Question: {state['cleaned_question']}

Context:
{context_text}

Answer:
{state['answer']}

Is this answer fully supported by the context?"""

    response = _client.chat.completions.create(
        model=CHAT_DEPLOYMENT_NAME,
        temperature=VALIDATOR_TEMPERATURE,
        max_tokens=VALIDATOR_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    result = response.choices[0].message.content.strip().upper()

    # Normalize — ensure only VALID or INVALID
    if "VALID" in result:
        validation_result = "VALID"
    else:
        validation_result = "INVALID"

    print(f"[Validator] ✅ Validation result: {validation_result}")
    return {**state, "validation_result": validation_result}
