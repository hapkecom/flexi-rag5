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
from common.utils.string_util import str_limit, merge_two_strings_with_with_overlap_detection
from langchain.text_splitter import TokenTextSplitter
import tiktoken

logger = logging.getLogger(__name__)

chunk_size_tokens = 500
chunk_overlap_tokens = 50


def split_single_document_into_parts_if_needed(doc: Document) -> List[Document]:
    """Split a single document into parts if needed.

    Args:
        doc: The document to split.

    Returns:
        A list of document parts.
    """
    text = doc.page_content

    # Check if the document is too large
    llm_model = "text-embedding-3-small" # doesn't need to be exact - we just need a rought guess here
    encoding = tiktoken.encoding_for_model(llm_model)
    token_count = len(encoding.encode(text))

    if token_count > chunk_size_tokens:
        # Split
        return split_single_document_into_parts(doc)
    else:
        # No need to split, but also add metadata
        doc_part = doc.metadata.get("part", "")
        doc_part_index = doc.metadata.get("part_index", 0)
        doc_split = doc #.pydantic_deep_copy()
        doc_split.metadata["part"] = doc_part
        doc_split.metadata["part_index"] = doc_part_index
        doc_split.metadata["sha256"] = sha256sum_str(doc_split.page_content)
        doc_split.metadata["size"] = len(doc_split.page_content)
        return [doc]


def split_single_document_into_parts(doc: Document) -> List[Document]:
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens, chunk_overlap=chunk_overlap_tokens
    )
    doc_splits = text_splitter.split_documents([doc])

    # add to metadata
    doc_part = doc.metadata.get("part", "")
    for i, doc_split in enumerate(doc_splits):
        # doesn't exist yet: doc_split.metadata["document_id"] = doc.metadata["id"]
        doc_split.metadata["part"] = f"{doc_part}/split/{i}"
        doc_split.metadata["part_index"] = i
        doc_split.metadata["sha256"] = sha256sum_str(doc_split.page_content)
        doc_split.metadata["size"] = len(doc_split.page_content)

    # add extended_page_content, i.e. the page_content + previous and next page_contents
    for i, doc_split in enumerate(doc_splits):
        # add previous page content
        extended_page_content = doc_split.page_content
        if i > 0:
            # previous page content exists
            extended_page_content = merge_two_strings_with_with_overlap_detection(
                doc_splits[i - 1].page_content,
                doc_split.page_content
            )
        # add next page content
        if i < len(doc_splits) - 1:
            # next page content exists
            extended_page_content = merge_two_strings_with_with_overlap_detection(
                extended_page_content,
                doc_splits[i + 1].page_content
            )
        # set the extended page content and futher metadata
        doc_split.metadata["extended_page_content"] = extended_page_content
        doc_split.metadata["extended_size"] = len(extended_page_content)
        doc_split.metadata["extended_sha256"] = sha256sum_str(extended_page_content)

    return doc_splits
