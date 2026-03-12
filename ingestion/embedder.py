from typing import List, Dict, Any
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_DEPLOYMENT_NAME,
    EMBEDDING_BATCH_SIZE,
)

# ── Initialize Azure OpenAI Client ──────────────────────────────────
_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates Azure OpenAI embeddings for each chunk.
    Processes in batches as defined in config.

    Args:
        chunks: List of chunk dicts from chunker.py

    Returns:
        Same list of chunk dicts with 'embedding' key added to each:
        {
            "chunk_id"  : str,
            "chunk_text": str,
            "embedding" : List[float],
            "metadata"  : dict
        }
    """
    total = len(chunks)
    print(f"[Embedder] Generating embeddings for {total} chunks in batches of {EMBEDDING_BATCH_SIZE}...")

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + EMBEDDING_BATCH_SIZE]
        batch_texts = [chunk["chunk_text"] for chunk in batch]

        response = _client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_NAME,
            input=batch_texts,
        )

        for i, embedding_obj in enumerate(response.data):
            chunks[batch_start + i]["embedding"] = embedding_obj.embedding

        print(f"[Embedder] Batch {batch_start // EMBEDDING_BATCH_SIZE + 1} complete "
              f"({min(batch_start + EMBEDDING_BATCH_SIZE, total)}/{total})")

    print(f"[Embedder] All embeddings generated successfully")
    return chunks
