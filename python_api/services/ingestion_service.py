import uuid
import os
import sys

def is_chroma_populated() -> bool:
    """
    Returns True if any of the domain collections have >0 chunks.
    Mirrors logic in Streamlit app's startup check.
    """
    try:
        import posthog
        posthog.disabled = True
        from ingestion.chroma_store import get_collection_stats
        stats = get_collection_stats()
        return any(count > 0 for count in stats.values())
    except Exception as e:
        print(f"[Startup] ChromaDB check failed: {e}")
        return False


def run_auto_ingestion() -> dict:
    """
    Runs your full ingestion pipeline using Azure Blob storage.
    Copied from Streamlit app's approach; no UI calls.
    """
    print("\n[Ingestion] Starting pipeline...")
    try:
        from azure.storage.blob import BlobServiceClient
        from config import BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME
        from ingestion.blob_reader import read_blob_file
        from ingestion.text_extractor import extract_text
        from ingestion.pii_masker import mask_pii
        from ingestion.domain_classifier import classify_domain
        from ingestion.chunker import chunk_document
        from ingestion.embedder import generate_embeddings
        from ingestion.chroma_store import store_chunks
        from ingestion.hash_tracker import (
            compute_hash,
            is_already_ingested,
            mark_as_ingested,
        )

        SUPPORTED_EXTENSIONS = {"pdf", "docx", "pptx", "txt"}

        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

        all_blobs = list(container_client.list_blobs())
        blob_names = [
            blob.name for blob in all_blobs
            if blob.name.rsplit(".", 1)[-1].lower() in SUPPORTED_EXTENSIONS
        ]
        print(f"[Ingestion] Found {len(blob_names)} supported documents")

        if not blob_names:
            return {"success": True, "ingested": 0, "skipped": 0, "failed": 0, "total": 0}

        ingested = skipped = failed = 0
        for blob_name in blob_names:
            try:
                file_bytes, extension = read_blob_file(blob_name)
                file_hash = compute_hash(file_bytes)

                if is_already_ingested(blob_name, file_hash):
                    skipped += 1
                    continue

                raw_text = extract_text(file_bytes, extension)
                if not raw_text.strip():
                    failed += 1
                    continue

                masked_text = mask_pii(raw_text)
                domain = classify_domain(masked_text)

                doc_id = str(uuid.uuid4())
                doc_name = blob_name.rsplit("/", 1)[-1]

                chunks = chunk_document(masked_text, doc_id, doc_name, domain)
                embedded_chunks = generate_embeddings(chunks)
                store_chunks(embedded_chunks, domain)

                mark_as_ingested(blob_name, file_hash)
                print(f"[Ingestion] ✅ {doc_name} → {domain}")
                ingested += 1

            except Exception as e:
                print(f"[Ingestion] ❌ {blob_name}: {str(e)}")
                failed += 1
                continue

        print(f"[Ingestion] Done — Ingested: {ingested} | Skipped: {skipped} | Failed: {failed}\n")
        return {
            "success": True,
            "ingested": ingested,
            "skipped": skipped,
            "failed": failed,
            "total": len(blob_names),
        }
    except Exception as e:
        print(f"[Ingestion] ❌ Error: {str(e)}")
        return {"success": False, "error": str(e)}