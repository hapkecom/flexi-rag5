
from typing import Optional, cast
from langchain.schema import Document


def get_document_source(doc: Document) -> Optional[str]:
    """
    Get the source of the document
    """

    if doc.metadata and "source" in doc.metadata:
        return cast(Optional[str], doc.metadata["source"])
    if doc.reference_url:
        return doc.reference_url
    return None
