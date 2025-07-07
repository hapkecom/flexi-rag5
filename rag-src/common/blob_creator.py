import os
from common.utils.hash_util import sha256sum_file

from typing import Any, Dict, Iterable, Iterator, Optional
from langchain_core.documents.base import Blob as Blob
from langchain_community.document_loaders.helpers import detect_file_encodings
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# Construct result Blob
def create_blob_from_local_file(url: str, 
                                file_path: str,
                                mimetype: Optional[str],
                                file_size: Optional[int]) -> Blob:

    # Create metadata of the blob
    if file_size is None:
        file_size = os.path.getsize(file_path)
    metadata: Dict[str, Any] = {
        "source": url,
        "file_size": file_size,
        "file_path": file_path,
        "file_sha256": sha256sum_file(file_path),
        "file_last_modified": datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    # Create blob
    blob = Blob.from_path(
        path = file_path,
        encoding = _guess_file_encoding(file_path),
        mime_type = mimetype,
        guess_type = True,
        metadata = metadata,
    )

    # Result
    return blob


def _guess_file_encoding(file_path: str) -> Optional[str]:
    try:
        detected_encodings = detect_file_encodings(file_path, timeout=30)
        if len(detected_encodings) > 0:
            # the most likely encoding is the first one
            return detected_encodings[0].encoding
        logger.info(f"Couldn't detecting encoding for '{file_path}'")

    except Exception as e:
        # Handle the exception if needed
        logger.warning(f"Error detecting encoding for '{file_path}': {e}")
        # You can log the error or raise an exception if necessary
    
    # Fallback to utf-8
    default_encoding = "utf-8"
    return default_encoding
