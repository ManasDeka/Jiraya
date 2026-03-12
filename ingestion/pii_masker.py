import re


# ── PII Patterns ────────────────────────────────────────────────────
PII_PATTERNS = [
    # Email
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),

    # Phone numbers (various formats)
    (r'\b(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b', '[PHONE]'),

    # SSN (US format)
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),

    # Credit/Debit card numbers (16 digit)
    (r'\b(?:\d{4}[\s\-]?){3}\d{4}\b', '[CARD_NUMBER]'),

    # Bank account numbers (8-18 digits)
    (r'\b\d{8,18}\b', '[BANK_ACCOUNT]'),

    # IFSC Code (India)
    (r'\b[A-Z]{4}0[A-Z0-9]{6}\b', '[IFSC_CODE]'),

    # Aadhaar number (India — 12 digits)
    (r'\b\d{4}\s\d{4}\s\d{4}\b', '[AADHAAR]'),

    # PAN Card (India)
    (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN]'),

    # IP Address
    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_ADDRESS]'),

    # Date of Birth patterns (DD/MM/YYYY or MM-DD-YYYY)
    (r'\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b', '[DOB]'),
]


def mask_pii(text: str) -> str:
    """
    Detects and masks PII entities in text using regex patterns.
    Original document in Blob remains untouched.
    Only masked text flows into classification and embedding.

    Args:
        text: Raw extracted document text

    Returns:
        masked_text with PII replaced by placeholders
    """
    masked_text = text

    for pattern, placeholder in PII_PATTERNS:
        masked_text = re.sub(pattern, placeholder, masked_text)

    original_len = len(text)
    masked_len = len(masked_text)
    print(f"[PIIMasker] Masking complete — Original: {original_len} chars | Masked: {masked_len} chars")

    return masked_text
