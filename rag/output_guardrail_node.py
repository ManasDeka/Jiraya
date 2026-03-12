"""
Output Guardrail Node
----------------------
Runs on the final generated answer BEFORE returning to user.
Checks for:
  1. Profanity in LLM output
  2. Accidental PII leakage in answer

If flagged → replaces answer with a safe fallback message.
Toggle: ENABLE_OUTPUT_GUARDRAIL in config.py
"""

import re
from rag.state import RAGState
from config import ENABLE_OUTPUT_GUARDRAIL

# ── Profanity List ───────────────────────────────────────────────────
_PROFANITY_LIST = [
    "damn", "hell", "crap", "shit", "fuck", "ass",
    "bastard", "bitch", "idiot", "stupid", "moron"
]

# ── PII Detection Patterns ───────────────────────────────────────────
_PII_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',   # Email
    r'\b\d{3}-\d{2}-\d{4}\b',                                    # SSN
    r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',                               # PAN
    r'\b\d{4}\s\d{4}\s\d{4}\b',                                  # Aadhaar
]


def _contains_profanity(text: str) -> bool:
    text_lower = text.lower()
    for word in _PROFANITY_LIST:
        if re.search(rf'\b{word}\b', text_lower):
            return True
    return False


def _contains_pii(text: str) -> bool:
    for pattern in _PII_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def output_guardrail_node(state: RAGState) -> RAGState:
    """
    Output Guardrail Node.
    Scans the final answer for profanity or PII leakage.
    Replaces flagged answers with a safe message.
    """

    if not ENABLE_OUTPUT_GUARDRAIL:
        print("[OutputGuardrail] Disabled — passing answer through")
        return {**state, "output_flagged": False}

    # If input guardrail already handled it → skip
    if state.get("guardrail_triggered"):
        print("[OutputGuardrail] Skipping — input guardrail already triggered")
        return {**state, "output_flagged": False}

    answer = state["answer"]

    # ── Profanity Check ──────────────────────────────────────────────
    if _contains_profanity(answer):
        print("[OutputGuardrail] ⚠️ Profanity detected in answer — flagging")
        return {
            **state,
            "answer": "The generated response was flagged by content policy. Please contact support.",
            "output_flagged": True,
        }

    # ── PII Leakage Check ────────────────────────────────────────────
    if _contains_pii(answer):
        print("[OutputGuardrail] ⚠️ PII detected in answer — flagging")
        return {
            **state,
            "answer": "The generated response contained sensitive information and has been blocked.",
            "output_flagged": True,
        }

    print("[OutputGuardrail] ✅ Answer passed output guardrail")
    return {**state, "output_flagged": False}
