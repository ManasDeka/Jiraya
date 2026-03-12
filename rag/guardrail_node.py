"""
Input Guardrail Node
---------------------
Runs on raw user query before anything else.
Performs:
  1. Profanity filtering — blocks abusive/offensive input
  2. PII detection      — warns user not to share personal data

If profanity detected → returns abusive message warning to user
If PII detected       → returns sensitive info warning to user
Clean query           → masks PII and flows into classifier
"""

import re
from rag.state import RAGState
from config import ENABLE_INPUT_GUARDRAIL

# ── Profanity Word List ──────────────────────────────────────────────
_PROFANITY_LIST = [
    "damn", "hell", "crap", "shit", "fuck", "ass",
    "bastard", "bitch", "idiot", "stupid", "moron",
    "dumb", "jerk", "loser", "retard", "piss",
    "dick", "cock", "pussy", "asshole", "wtf",
]

# ── PII Detection Patterns ────────────────────────────────────────────
_PII_DETECTION_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email address"),
    (r'\b(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b', "phone number"),
    (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),
    (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', "PAN number"),
    (r'\b\d{4}\s\d{4}\s\d{4}\b', "Aadhaar number"),
    (r'\b[A-Z]{4}0[A-Z0-9]{6}\b', "IFSC code"),
    (r'\b(?:\d{4}[\s\-]?){3}\d{4}\b', "card number"),
]

# ── PII Masking Patterns (for clean queries) ─────────────────────────
_PII_MASK_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
    (r'\b(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b', '[PHONE]'),
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
    (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN]'),
    (r'\b\d{4}\s\d{4}\s\d{4}\b', '[AADHAAR]'),
    (r'\b[A-Z]{4}0[A-Z0-9]{6}\b', '[IFSC]'),
    (r'\b(?:\d{4}[\s\-]?){3}\d{4}\b', '[CARD]'),
]


def _check_profanity(text: str) -> bool:
    """Returns True if text contains any profanity word."""
    text_lower = text.lower()
    for word in _PROFANITY_LIST:
        if re.search(rf'\b{word}\b', text_lower):
            return True
    return False


def _check_pii(text: str) -> tuple[bool, str]:
    """
    Checks if text contains PII.

    Returns:
        (True, pii_type) if PII found
        (False, "")      if clean
    """
    for pattern, pii_type in _PII_DETECTION_PATTERNS:
        if re.search(pattern, text):
            return True, pii_type
    return False, ""


def _mask_pii(text: str) -> str:
    """Masks PII entities in clean query text."""
    for pattern, placeholder in _PII_MASK_PATTERNS:
        text = re.sub(pattern, placeholder, text)
    return text


def guardrail_node(state: RAGState) -> RAGState:
    """
    Input Guardrail Node.

    Flow:
      1. If guardrail disabled → pass through as-is
      2. Check profanity → block with professional warning
      3. Check PII → block with sensitive info warning
      4. Clean query → mask any residual PII → pass to classifier
    """

    if not ENABLE_INPUT_GUARDRAIL:
        print("[InputGuardrail] Disabled — passing query through")
        return {**state, "cleaned_question": state["question"]}

    question = state["question"]

    # ── Step 1: Profanity Check ───────────────────────────────────────
    if _check_profanity(question):
        print("[InputGuardrail] Profanity detected — blocking query")
        return {
            **state,
            "cleaned_question": "",
            "guardrail_triggered": True,
            "answer": (
                "⚠️ Your message contains abusive or inappropriate language. "
                "This platform is for professional enterprise use only. "
                "Please rephrase your question respectfully."
            ),
        }

    # ── Step 2: PII Detection ─────────────────────────────────────────
    has_pii, pii_type = _check_pii(question)
    if has_pii:
        print(f"[InputGuardrail] PII detected ({pii_type}) — blocking query")
        return {
            **state,
            "cleaned_question": "",
            "guardrail_triggered": True,
            "answer": (
                f"⚠️ Your query appears to contain personal sensitive information ({pii_type}). "
                "Please do not share personal data such as email addresses, phone numbers, "
                "ID numbers, or financial details in your queries. "
                "Rephrase your question without personal information."
            ),
        }

    # ── Step 3: Clean Query — mask any residual PII and continue ─────
    cleaned = _mask_pii(question)
    print("[InputGuardrail] Query clean — proceeding to classifier")

    return {
        **state,
        "cleaned_question": cleaned,
        "guardrail_triggered": False,
    }
