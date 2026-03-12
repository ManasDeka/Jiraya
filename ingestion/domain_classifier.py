"""
Domain Classifier — Reusable Function
--------------------------------------
This module is designed to be used independently.
Works for both:
  1. Ingestion pipeline  — classify full document text
  2. RAG query routing   — classify user query text

Usage:
    from ingestion.domain_classifier import classify_domain
    domain = classify_domain("your text or query here")
    # Returns one of: "HR", "IT", "Finance", "Operations"
"""

from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    CHAT_DEPLOYMENT_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SUPPORTED_DOMAINS,
)

# ── Initialize Azure OpenAI Client ──────────────────────────────────
_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a domain classification assistant for an enterprise document system.
Your job is to classify the given text into exactly one of the following domains:
HR, IT, Finance, Operations

Rules:
- Return ONLY the domain name. No explanation. No punctuation.
- Choose the most relevant domain based on the content.
- If uncertain, pick the closest matching domain.

Valid outputs: HR | IT | Finance | Operations"""


def classify_domain(text: str) -> str:
    """
    Classifies text into one of the supported enterprise domains.

    This function is fully reusable:
    - Pass full document text during ingestion
    - Pass user query text during RAG routing

    Args:
        text: Any text string (document content or user query)

    Returns:
        domain: One of 'HR', 'IT', 'Finance', 'Operations'

    Raises:
        ValueError: If LLM returns an unexpected domain
    """
    # Truncate to avoid token overflow for large documents
    # Classification only needs a representative sample
    truncated_text = text[:4000] if len(text) > 4000 else text

    response = _client.chat.completions.create(
        model=CHAT_DEPLOYMENT_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this text:\n\n{truncated_text}"},
        ],
    )

    raw_output = response.choices[0].message.content.strip()

    # Normalize and validate
    domain = raw_output.strip().capitalize()
    # Handle exact case matching for supported domains
    for supported in SUPPORTED_DOMAINS:
        if supported.lower() == domain.lower():
            print(f"[DomainClassifier] Classified as: {supported}")
            return supported

    # Fallback — if LLM returns unexpected value, raise clearly
    raise ValueError(
        f"[DomainClassifier] Unexpected domain returned: '{raw_output}'. "
        f"Expected one of: {SUPPORTED_DOMAINS}"
    )
