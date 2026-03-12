import re
import uuid
from typing import List, Dict, Any
from config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, DEFAULT_DOC_VERSION


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimator: ~1 token per 4 characters.
    Avoids tiktoken dependency for enterprise simplicity.
    """
    return len(text) // 4


def _split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_page_number(text: str, char_position: int) -> int:
    """
    Attempts to find the nearest [PAGE X] marker before a given char position.
    Returns page number or 1 as default.
    """
    page_markers = [(m.start(), int(m.group(1))) for m in re.finditer(r'\[PAGE (\d+)\]', text)]
    current_page = 1
    for pos, page_num in page_markers:
        if pos <= char_position:
            current_page = page_num
        else:
            break
    return current_page


def chunk_document(
    text: str,
    doc_id: str,
    doc_name: str,
    domain: str,
) -> List[Dict[str, Any]]:
    """
    Custom chunker — splits document into token-aware chunks
    with sentence boundary respect and overlap.

    Args:
        text     : Full masked document text
        doc_id   : Unique document identifier
        doc_name : Original file name
        domain   : Classified domain (HR / IT / Finance / Operations)

    Returns:
        List of chunk dicts, each containing:
        {
            "chunk_id"    : str,
            "chunk_text"  : str,
            "metadata"    : {
                "doc_id"      : str,
                "doc_name"    : str,
                "domain"      : str,
                "chunk_id"    : str,
                "page_number" : int,
                "version"     : str,
            }
        }
    """
    sentences = _split_into_sentences(text)
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    chunk_index = 0

    # Track approximate character position for page detection
    processed_chars = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If adding this sentence exceeds chunk size → finalize current chunk
        if current_token_count + sentence_tokens > CHUNK_SIZE_TOKENS and current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            page_number = _extract_page_number(text, processed_chars)

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "domain": domain,
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "version": DEFAULT_DOC_VERSION,
                },
            })

            chunk_index += 1

            # Apply overlap — keep last N tokens worth of sentences
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = _estimate_tokens(s)
                if overlap_tokens + s_tokens <= CHUNK_OVERLAP_TOKENS:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break

            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens
        processed_chars += len(sentence)

    # Final remaining chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunk_id = f"{doc_id}_chunk_{chunk_index}"
        page_number = _extract_page_number(text, processed_chars)

        chunks.append({
            "chunk_id": chunk_id,
            "chunk_text": chunk_text,
            "metadata": {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "domain": domain,
                "chunk_id": chunk_id,
                "page_number": page_number,
                "version": DEFAULT_DOC_VERSION,
            },
        })

    print(f"[Chunker] '{doc_name}' → {len(chunks)} chunks created")
    return chunks
