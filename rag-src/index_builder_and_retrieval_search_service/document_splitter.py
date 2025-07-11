from typing import TYPE_CHECKING
from typing import (
    Any,
    List,
)
import mimetypes
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import logging

from common.utils.hash_util import sha256sum_str
from common.utils.string_util import str_limit
from langchain.text_splitter import TokenTextSplitter
import tiktoken

logger = logging.getLogger(__name__)

chunk_size=500
chunk_overlap=50


def split_single_document_into_parts_if_needed(doc: Document) -> List[Document]:
    """Split a single document into parts if needed.

    Args:
        doc: The document to split.
        max_part_size: The maximum size of each part.
        max_part_overlap: The maximum overlap between parts.

    Returns:
        A list of document parts.
    """
    text = doc.page_content

    # Check if the document is too large
    llm_model = "text-embedding-3-small" # doesn't need to be exact - we just need a rought guess here
    encoding = tiktoken.encoding_for_model(llm_model)
    token_count = len(encoding.encode(text))

    if token_count > chunk_size:
        # Split
        return split_single_document_into_parts(doc)
    else:
        # No need to split, but also add metadata
        doc_split = doc #.pydantic_deep_copy()
        doc_split.metadata["part_index"] = 0
        doc_split.metadata["sha256"] = sha256sum_str(doc_split.page_content)
        doc_split.metadata["size"] = len(doc_split.page_content)
        return [doc]


def split_single_document_into_parts(doc: Document) -> List[Document]:
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents([doc])

    # add to metadata
    for i, doc_split in enumerate(doc_splits):
        # doesn't exist yet: doc_split.metadata["document_id"] = doc.metadata["id"]
        doc_split.metadata["part_index"] = i
        doc_split.metadata["sha256"] = sha256sum_str(doc_split.page_content)
        doc_split.metadata["size"] = len(doc_split.page_content)

    return doc_splits
