from langchain_core.documents.base import Blob
from typing import TYPE_CHECKING
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from pathlib import PurePath
PathLike = Union[str, PurePath]
import logging

logger = logging.getLogger(__name__)

#
# Guess the mime-type of a Blob if no explicit mime-type is available
#

def guess_mime_type(blob: Blob) -> Optional[str]:
    """
    Guess the mime-type of a Blob if no explicit mime-type is available
    - file based on its extension
    - text based on its content
    """


    # TODO: check:   mimetypes.guess_type(path)

    # Check if the blob has a mime-type
    if blob.mimetype is not None:
        return blob.mimetype

    # Check if the blob has an URL
    url = blob.source
    if url is None:
        mime_type = guess_mime_type_from_url(url)
        if mime_type is not None:
            return mime_type

    # Check if the blob has a local file path
    path = blob.path
    if path is not None:
        suffix = path.suffix
        if suffix is not None:
            mime_type = guess_mime_type_from_url(suffix)
            if mime_type is not None:
                return mime_type

    # Check if the blob has a content
    mime_type = guess_mime_type_from_content(blob)
    if mime_type is not None:
        return mime_type

    # If no mime-type is available, raise an error
    #raise ValueError(f"Blob {blob} does not have a mime-type and no path to guess from.")
    return None

def guess_mime_type_from_url(url: str) -> Optional[str]:
    if url is None:
        return None

    # URL preparation
    if "?" in url:
        # Remove query parameters
        url = url.split("?")[0]
    url = url.lower()

    # Check different suffixes
    if url.endswith(".pdf"):
        return "application/pdf"
    if url.endswith(".txt"):
        return "text/plain"
    if url.endswith(".html"):
        return "text/html"
    if url.endswith(".json"):
        return "application/json"
    if url.endswith(".csv"):
        return "text/csv"
    if url.endswith(".xml"):
        return "application/xml"

    # Check for programming languages
    if url.endswith(".java"):
        return "text/x-java-source"
    if url.endswith(".py"):
        return "text/x-python"
    if url.endswith(".js"):
        return "text/javascript"
    if url.endswith(".ts"):
        return "text/typescript"
    if url.endswith(".css"):
        return "text/css"
    if url.endswith(".sh"):
        return "text/x-shellscript"
    if url.endswith(".bash"):
        return "text/x-shellscript"

    # If no mime-type is available, return None
    return None

def guess_mime_type_from_content(blob: Blob) -> str:
    # TO IMPLEMENT
    logger.warning(f"guess_mime_type_from_content() is not implemented yet - for blob.source: '{blob.source}'")
    return None
