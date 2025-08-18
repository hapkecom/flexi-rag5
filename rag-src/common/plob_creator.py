from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from model.plob import Plob
import shortuuid
import logging

logger = logging.getLogger(__name__)


def create_plob_with_metadata_of_blob(blob: Blob) -> Plob:
    """Create a plob with metadata from the blob. Without documents yet."""

    # Create a new plob object
    plob = Plob(
        id = "plob-"+str(shortuuid.uuid()[:8]),
        url = blob.source,
        media_type = blob.mimetype,
        file_path = blob.metadata.get("file_path", "-"),
        file_size = blob.metadata.get("file_size", 0),
        file_sha256 = blob.metadata.get("file_sha256", "-"), 
        file_last_modified = blob.metadata.get("file_last_modified", "-"),
        documents = [],
    )

    # Copy metadata from the blob to the plob
    for key, value in blob.metadata.items():
        if key not in plob.metadata and key not in ["file_path", "file_size", "file_sha256", "file_last_modified"]:
            plob.metadata[key] = value

    return plob


def create_virtual_plob(name: str) -> Plob:
    """Create a plob with metadata from the blob. Without documents yet."""

    # Create a new plob object
    plob = Plob(
        id = "plob-"+str(shortuuid.uuid()[:8]),
        url = name,
        media_type = "application/unkown", # "application/octet-stream"
        file_path = "-",
        file_size = 0,
        file_sha256 = "-", 
        file_last_modified = "-",
        documents = [],
    )

    return plob
