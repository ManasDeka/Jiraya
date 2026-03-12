import io
from azure.storage.blob import BlobServiceClient
from config import BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME


def read_blob_file(blob_name: str) -> tuple[bytes, str]:
    """
    Downloads a file from Azure Blob Storage.

    Args:
        blob_name: Full path/name of the blob (e.g., 'hr/policy.pdf')

    Returns:
        Tuple of (file_bytes, file_extension)
        e.g., (b'...', 'pdf')
    """
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)

    download_stream = blob_client.download_blob()
    file_bytes = download_stream.readall()

    # Extract extension from blob name
    extension = blob_name.rsplit(".", 1)[-1].lower()

    print(f"[BlobReader] Successfully downloaded: {blob_name} ({len(file_bytes)} bytes)")
    return file_bytes, extension
