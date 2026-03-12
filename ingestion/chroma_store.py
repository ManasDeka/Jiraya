"""
ChromaDB Store
---------------
Stores embedded chunks into domain-specific ChromaDB collections.
All 4 collections persist inside chroma_db/chroma.sqlite3

Collections:
    hr_collection
    it_collection
    finance_collection
    operations_collection
"""

import posthog
posthog.disabled = True

import chromadb
from chromadb.config import Settings
from config import (
    CHROMA_DB_PATH,
    CHROMA_HR_COLLECTION,
    CHROMA_IT_COLLECTION,
    CHROMA_FINANCE_COLLECTION,
    CHROMA_OPERATIONS_COLLECTION,
)

# ── Domain → Collection Name Mapping ────────────────────────────────
DOMAIN_COLLECTION_MAP = {
    "HR":         CHROMA_HR_COLLECTION,
    "IT":         CHROMA_IT_COLLECTION,
    "Finance":    CHROMA_FINANCE_COLLECTION,
    "Operations": CHROMA_OPERATIONS_COLLECTION,
}

# ── Singleton Client ─────────────────────────────────────────────────
_chroma_client = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a singleton ChromaDB PersistentClient.
    Auto-creates chroma_db/ folder and chroma.sqlite3.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        print(f"[ChromaStore] Connected to ChromaDB at: {CHROMA_DB_PATH}")
    return _chroma_client


def get_or_create_collection(domain: str):
    """
    Gets existing collection or creates it if not present.

    Args:
        domain : One of HR, IT, Finance, Operations

    Returns:
        ChromaDB Collection object
    """
    client          = get_chroma_client()
    collection_name = DOMAIN_COLLECTION_MAP.get(domain)

    if not collection_name:
        raise ValueError(f"[ChromaStore] Unknown domain: '{domain}'")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def store_chunks(embedded_chunks: list, domain: str) -> int:
    """
    Stores embedded chunks into the appropriate domain collection.

    Handles both chunk key formats:
        "chunk_text"  → from chunker/embedder pipeline
        "text"        → alternate key format

    Args:
        embedded_chunks : List of chunk dicts containing:
                          chunk_id, chunk_text/text, embedding, metadata
        domain          : HR / IT / Finance / Operations

    Returns:
        Number of chunks successfully stored
    """
    if not embedded_chunks:
        print(f"[ChromaStore] No chunks to store for domain: {domain}")
        return 0

    collection = get_or_create_collection(domain)

    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for chunk in embedded_chunks:

        # ── Get chunk ID ─────────────────────────────────────────────
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            print(f"[ChromaStore] ⚠️ Chunk missing chunk_id — skipping")
            continue

        # ── Get chunk text — handles both key names ──────────────────
        chunk_text = chunk.get("chunk_text") or chunk.get("text")
        if not chunk_text:
            print(f"[ChromaStore] ⚠️ Chunk {chunk_id} has no text — skipping")
            continue

        # ── Get embedding ─────────────────────────────────────────────
        embedding = chunk.get("embedding")
        if not embedding:
            print(f"[ChromaStore] ⚠️ Chunk {chunk_id} has no embedding — skipping")
            continue

        # ── Get metadata — flatten if nested ─────────────────────────
        metadata = chunk.get("metadata", {})

        # ChromaDB requires metadata values to be str/int/float/bool only
        # Flatten any nested structures to strings
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)

        # ── Also store top-level fields in metadata if not already ───
        for field in ["doc_id", "doc_name", "domain"]:
            if field not in clean_metadata and field in chunk:
                clean_metadata[field] = str(chunk[field])

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk_text)
        metadatas.append(clean_metadata)

    if not ids:
        print(f"[ChromaStore] No valid chunks to store for domain: {domain}")
        return 0

    # ── Upsert to ChromaDB ───────────────────────────────────────────
    # Upsert = insert new + update existing (by chunk_id)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[ChromaStore] ✅ Stored {len(ids)} chunks → '{collection.name}' collection")
    return len(ids)


def get_collection_stats() -> dict:
    """
    Returns chunk count for all 4 domain collections.
    Used by startup check and sidebar live stats.

    Returns:
        dict: { "HR": int, "IT": int, "Finance": int, "Operations": int }
    """
    client = get_chroma_client()
    stats  = {}

    for domain, col_name in DOMAIN_COLLECTION_MAP.items():
        try:
            col           = client.get_collection(col_name)
            stats[domain] = col.count()
        except Exception:
            stats[domain] = 0

    return stats
