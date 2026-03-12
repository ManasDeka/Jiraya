"""
Document Uploader
------------------
Handles:
  1. File upload from Streamlit UI
  2. Push file to Azure Blob Storage
  3. Trigger ingestion pipeline silently
  4. Return result (success/failure) to UI
"""

import logging
from azure.storage.blob import BlobServiceClient
from config import BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME

# ── Suppress all pipeline logs from showing in UI ───────────────────
logging.disable(logging.CRITICAL)

# ── Supported formats ────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {"pdf", "docx", "pptx", "txt"}


def upload_to_blob(file_bytes: bytes, filename: str) -> str:
    """
    Uploads file bytes to Azure Blob Storage.

    Args:
        file_bytes : Raw file bytes from Streamlit uploader
        filename   : Original filename

    Returns:
        blob_name : Full path stored in blob container
    """
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

    blob_name = filename
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(file_bytes, overwrite=True)

    return blob_name


def run_ingestion_silent(blob_name: str) -> dict:
    """
    Runs the full ingestion pipeline silently.
    Suppresses all print/log output so UI stays clean.

    Args:
        blob_name : Blob path of uploaded file

    Returns:
        result dict with keys: success, domain, chunks, doc_id, error
    """
    import sys
    import os
    import uuid as _uuid

    from ingestion.blob_reader import read_blob_file
    from ingestion.text_extractor import extract_text
    from ingestion.pii_masker import mask_pii
    from ingestion.domain_classifier import classify_domain
    from ingestion.chunker import chunk_document
    from ingestion.embedder import generate_embeddings
    from ingestion.chroma_store import store_chunks
    from ingestion.hash_tracker import compute_hash, is_already_ingested, mark_as_ingested

    old_stdout = sys.stdout
    result = {}

    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

        # ── Read file ────────────────────────────────────────────────
        file_bytes, extension = read_blob_file(blob_name)

        # ── Hash check ───────────────────────────────────────────────
        file_hash = compute_hash(file_bytes)
        if is_already_ingested(blob_name, file_hash):
            result = {
                "success": True,
                "skipped": True,
                "message": "Document already indexed (no changes detected)",
                "domain": "N/A",
                "chunks": 0,
            }
            return result

        # ── Full pipeline ────────────────────────────────────────────
        raw_text = extract_text(file_bytes, extension)
        masked_text = mask_pii(raw_text)
        domain = classify_domain(masked_text)
        doc_id = str(_uuid.uuid4())
        doc_name = blob_name.rsplit("/", 1)[-1]

        chunks = chunk_document(
            text=masked_text,
            doc_id=doc_id,
            doc_name=doc_name,
            domain=domain,
        )

        embedded_chunks = generate_embeddings(chunks)
        store_chunks(embedded_chunks, domain)
        mark_as_ingested(blob_name, file_hash)

        result = {
            "success": True,
            "skipped": False,
            "domain": domain,
            "chunks": len(embedded_chunks),
            "doc_id": doc_id,
        }
        return result

    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
        }
        return result

    finally:
        # ✅ Always restores stdout regardless of success/failure
        sys.stdout = old_stdout


def handle_upload(uploaded_file) -> dict:
    """
    Full upload handler called from Streamlit UI.

    Args:
        uploaded_file : Streamlit UploadedFile object

    Returns:
        result dict
    """
    filename = uploaded_file.name
    extension = filename.rsplit(".", 1)[-1].lower()

    # ── Validate format ──────────────────────────────────────────────
    if extension not in SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported format '.{extension}'. Allowed: PDF, DOCX, PPTX, TXT",
        }

    file_bytes = uploaded_file.read()

    # ── Upload to Azure Blob ─────────────────────────────────────────
    try:
        blob_name = upload_to_blob(file_bytes, filename)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to upload to Azure Blob: {str(e)}",
        }

    # ── Run ingestion silently ───────────────────────────────────────
    result = run_ingestion_silent(blob_name)
    result["filename"] = filename
    return result
