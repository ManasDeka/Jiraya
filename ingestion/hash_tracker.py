"""
Hash Tracker — Prevents Duplicate Ingestion
---------------------------------------------
Stores MD5 hashes of already-ingested documents.
If a file's content hash already exists → skip ingestion.
If file is new or modified → process and update hash.

Hash store: ingested_hashes.json (auto-created in project root)
"""

import hashlib
import json
import os

HASH_STORE_PATH = "ingested_hashes.json"


def _load_hash_store() -> dict:
    """Loads existing hash store from disk. Returns empty dict if not found."""
    if os.path.exists(HASH_STORE_PATH):
        with open(HASH_STORE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_hash_store(store: dict) -> None:
    """Persists hash store to disk."""
    with open(HASH_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)


def compute_hash(file_bytes: bytes) -> str:
    """
    Computes MD5 hash of file bytes.

    Args:
        file_bytes: Raw bytes of the document

    Returns:
        MD5 hex digest string
    """
    return hashlib.md5(file_bytes).hexdigest()


def is_already_ingested(blob_name: str, file_hash: str) -> bool:
    """
    Checks if this exact file content was already ingested.

    Args:
        blob_name : Blob path used as key
        file_hash : MD5 hash of current file bytes

    Returns:
        True  → already ingested, skip
        False → new or modified file, process it
    """
    store = _load_hash_store()
    stored_hash = store.get(blob_name)
    return stored_hash == file_hash


def mark_as_ingested(blob_name: str, file_hash: str) -> None:
    """
    Saves the hash of a successfully ingested document.

    Args:
        blob_name : Blob path used as key
        file_hash : MD5 hash of the file bytes
    """
    store = _load_hash_store()
    store[blob_name] = file_hash
    _save_hash_store(store)
    print(f"[HashTracker] Hash saved for: {blob_name}")
