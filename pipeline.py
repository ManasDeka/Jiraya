"""
Ingestion Pipeline — Bulk Manual Trigger
------------------------------------------
Lists all documents from Azure Blob Storage and ingests
each one into the appropriate ChromaDB domain collection.

Hashing:
  - Skips files already ingested with identical content
  - Re-ingests if file content has changed (new hash)

Usage:
    python pipeline.py
"""

import uuid
from azure.storage.blob import BlobServiceClient

from config import BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME
from ingestion.blob_reader import read_blob_file
from ingestion.text_extractor import extract_text
from ingestion.pii_masker import mask_pii
from ingestion.domain_classifier import classify_domain
from ingestion.chunker import chunk_document
from ingestion.embedder import generate_embeddings
from ingestion.chroma_store import store_chunks
from ingestion.hash_tracker import compute_hash, is_already_ingested, mark_as_ingested

# ── Supported File Extensions ────────────────────────────────────────
SUPPORTED_EXTENSIONS = {"pdf", "docx", "pptx", "txt"}


def list_all_blobs() -> list[str]:
    """
    Lists all supported document blobs in the Azure container.

    Returns:
        List of blob names (paths) with supported extensions
    """
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

    blob_names = []
    for blob in container_client.list_blobs():
        extension = blob.name.rsplit(".", 1)[-1].lower()
        if extension in SUPPORTED_EXTENSIONS:
            blob_names.append(blob.name)

    print(f"[Pipeline] Found {len(blob_names)} supported document(s) in container\n")
    return blob_names


def run_ingestion(blob_name: str) -> None:
    """
    Runs the full ingestion pipeline for a single blob document.
    Skips processing if content hash already exists.

    Args:
        blob_name: Full blob path (e.g., 'hr/policy.pdf')
    """
    print(f"\n{'='*60}")
    print(f"  PROCESSING: {blob_name}")
    print(f"{'='*60}")

    # ── Step 1: Read from Blob ───────────────────────────────────────
    print("\n>> STEP 1: Reading from Azure Blob Storage...")
    file_bytes, extension = read_blob_file(blob_name)

    # ── Hash Check — Skip if Already Ingested ────────────────────────
    file_hash = compute_hash(file_bytes)

    if is_already_ingested(blob_name, file_hash):
        print(f"[Pipeline] SKIPPED — '{blob_name}' already ingested with same content")
        print(f"{'='*60}\n")
        return

    print(f"[Pipeline] New or modified file detected — proceeding with ingestion")

    # ── Step 2: Extract Text ─────────────────────────────────────────
    print("\n>> STEP 2: Extracting text...")
    raw_text = extract_text(file_bytes, extension)

    # ── Step 3: PII Masking ──────────────────────────────────────────
    print("\n>> STEP 3: Masking PII...")
    masked_text = mask_pii(raw_text)

    # ── Step 4: Domain Classification ───────────────────────────────
    print("\n>> STEP 4: Classifying domain...")
    domain = classify_domain(masked_text)

    # ── Step 5: Chunking ─────────────────────────────────────────────
    print("\n>> STEP 5: Chunking document...")
    doc_id = str(uuid.uuid4())
    doc_name = blob_name.rsplit("/", 1)[-1]
    chunks = chunk_document(
        text=masked_text,
        doc_id=doc_id,
        doc_name=doc_name,
        domain=domain,
    )

    # ── Step 6: Generate Embeddings ──────────────────────────────────
    print("\n>> STEP 6: Generating embeddings...")
    embedded_chunks = generate_embeddings(chunks)

    # ── Step 7: Store in ChromaDB ────────────────────────────────────
    print("\n>> STEP 7: Storing in ChromaDB...")
    store_chunks(embedded_chunks, domain)

    # ── Save Hash After Successful Ingestion ─────────────────────────
    mark_as_ingested(blob_name, file_hash)

    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE")
    print(f"  Document : {doc_name}")
    print(f"  Domain   : {domain}")
    print(f"  Chunks   : {len(embedded_chunks)}")
    print(f"  Doc ID   : {doc_id}")
    print(f"{'='*60}\n")


def main() -> None:
    """
    Entry point — lists all blobs and ingests each one.
    Automatically skips files already ingested with same content.
    """
    print("\n" + "="*60)
    print("   ENTERPRISE DOCUMENT INGESTION PIPELINE")
    print("="*60)

    blob_names = list_all_blobs()

    if not blob_names:
        print("[Pipeline] No supported documents found in blob container. Exiting.")
        return

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for blob_name in blob_names:
        try:
            # Peek at bytes just to compute hash before full processing
            file_bytes, _ = read_blob_file(blob_name)
            file_hash = compute_hash(file_bytes)

            if is_already_ingested(blob_name, file_hash):
                print(f"[Pipeline] SKIPPED — {blob_name} (already ingested, content unchanged)")
                skipped_count += 1
                continue

            run_ingestion(blob_name)
            success_count += 1

        except Exception as e:
            print(f"[Pipeline] FAILED — {blob_name} | Error: {e}")
            failed_count += 1
            continue  # Don't stop entire pipeline for one bad file

    # ── Final Summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("   PIPELINE RUN SUMMARY")
    print("="*60)
    print(f"  Total Found  : {len(blob_names)}")
    print(f"  Ingested     : {success_count}")
    print(f"  Skipped      : {skipped_count}")
    print(f"  Failed       : {failed_count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
