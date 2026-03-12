"""
Retriever Node
---------------
Queries the correct ChromaDB domain collection.
Uses Azure OpenAI to embed the user query.
Dynamic top_k:
  - retry_count == 0 → RAG_TOP_K_INITIAL (5)
  - retry_count > 0  → RAG_TOP_K_RETRY (10)
"""

import posthog
posthog.disabled = True

import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI

from rag.state import RAGState
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_DEPLOYMENT_NAME,
    CHROMA_HR_COLLECTION,
    CHROMA_IT_COLLECTION,
    CHROMA_FINANCE_COLLECTION,
    CHROMA_OPERATIONS_COLLECTION,
    RAG_TOP_K_INITIAL,
    RAG_TOP_K_RETRY,
)

# ── Azure OpenAI Embedding Client ────────────────────────────────────
_embedding_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# ── ChromaDB Client ──────────────────────────────────────────────────
_chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False),
)

# ── Domain → Collection Map ──────────────────────────────────────────
_COLLECTION_MAP = {
    "HR": CHROMA_HR_COLLECTION,
    "IT": CHROMA_IT_COLLECTION,
    "Finance": CHROMA_FINANCE_COLLECTION,
    "Operations": CHROMA_OPERATIONS_COLLECTION,
}


def _embed_query(text: str) -> list:
    """Generates embedding for the user query using Azure OpenAI."""
    response = _embedding_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT_NAME,
        input=[text],
    )
    return response.data[0].embedding


def retriever_node(state: RAGState) -> RAGState:
    """
    Retriever Node.
    Embeds the cleaned query and retrieves top_k chunks
    from the domain-specific ChromaDB collection.
    """

    if state.get("guardrail_triggered"):
        print("[Retriever] Skipping — guardrail triggered")
        return state

    domain = state["domain"]
    query = state["cleaned_question"]
    retry_count = state["retry_count"]

    # Dynamic top_k based on retry
    top_k = RAG_TOP_K_INITIAL if retry_count == 0 else RAG_TOP_K_RETRY
    print(f"[Retriever] Domain: {domain} | top_k: {top_k} | Attempt: {retry_count + 1}")

    # Get correct collection
    collection_name = _COLLECTION_MAP.get(domain)
    if not collection_name:
        raise ValueError(f"[Retriever] Unknown domain: {domain}")

    collection = _chroma_client.get_collection(name=collection_name)

    # Embed query
    query_embedding = _embed_query(query)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    chunks = [
        {
            "text": doc,
            "metadata": meta,
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    print(f"[Retriever] ✅ Retrieved {len(chunks)} chunks from '{collection_name}'")
    return {**state, "retrieved_chunks": chunks}
