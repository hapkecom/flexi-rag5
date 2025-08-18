from typing import TYPE_CHECKING
from typing import (
    Any,
    List,
)
from common.utils.hash_util import sha256sum_str
import logging
import asyncio
from langchain_core.documents import Document
from .document_splitter import split_single_document_into_parts_if_needed
from .document_summarizer import summarize_text
from common.service.configloader import deep_get, settings


logger = logging.getLogger(__name__)

include_summary_in_search_index = deep_get(settings, "config.rag_indexing.include_summary_in_search_index", default_value=False)
include_summary_in_search_results = deep_get(settings, "config.rag_indexing.include_summary_in_search_results", default_value=False)

#
# Put all logic together to split a document into parts here and to summarize
#

def improve_and_split_single_document_into_parts(doc: Document, logging_prefix: str = "") -> List[Document]:
    """
    Split a single document into parts if needed,
    and improve / enrich with additional LLM-generated summaries.
    """

    doc_results = []

    # Get the document parts
    doc_splits = split_single_document_into_parts_if_needed(doc)
    doc_results.extend(doc_splits)
    logger.info(f"{logging_prefix}DONE: Split document into {len(doc_splits)} document / parts: {doc.metadata.get('title', 'No title')}")

    # Optionally, caclulate and index summaries
    if include_summary_in_search_index:
        # Get summaries for each part
        logger.info(f"{logging_prefix}  Start adding about {len(doc_splits)} summaries for document: {doc.metadata.get('title', 'No title')} ...")
        summaries_of_doc_splits = [get_summary_document(doc_split) for doc_split in doc_splits]
        # Remove None summaries
        summaries_of_doc_splits = [summary for summary in summaries_of_doc_splits if summary is not None]
        # Handle metadata
        doc_results.extend(summaries_of_doc_splits)
        logger.info(f"{logging_prefix}  DONE: Added {len(summaries_of_doc_splits)} summaries for document: {doc.metadata.get('title', 'No title')}")

        # Join summaries and resplit (recursively if needed)
        if include_summary_in_search_results and (len(summaries_of_doc_splits) > 1):
            # Multiple summaries exist: join them and resplit and re-summarzed them
            logger.info(f"{logging_prefix}  Start RECURSION with adding joined summaries of document splits and resplitting them: {doc.metadata.get('title', 'No title')} ...")
            joined_summaries_doc = join_documents(summaries_of_doc_splits, optional_source_anker_to_add='remix')
            joined_summaries_doc_resplitted = improve_and_split_single_document_into_parts(joined_summaries_doc, logging_prefix="        ")
            doc_results.extend(joined_summaries_doc_resplitted)
            logger.info(f"{logging_prefix}  DONE RECURSION: Added {len(joined_summaries_doc_resplitted)} resplit joined summaries of document: {doc.metadata.get('title', 'No title')}")
    else:
        if include_summary_in_search_results:
            logger.warning(f"{logging_prefix}  WARNING: include_summary_in_search_results=True, but include_summary_in_search_index=False. This will not work as expected. Please check your configuration.")

    return doc_results



def join_documents(documents: List[Document], optional_source_anker_to_add: str | None) -> Document:
    # join summaries into a single (temporary) document
    joined_summary = Document(
        page_content="\n\n".join([doc.page_content for doc in documents]),
    )
    # add metadata, mainly from the first document
    joined_summary = enrich_document_from_parent_document(joined_summary, documents[0], "join")
    return joined_summary


#
# Summary generation
#

def get_summary_document(doc: Document) -> Document | None:
    """
    Get a document that contains the summary of the of the provided.
    """
    # summarize into a single (temporary) document
    original_page_content = doc.page_content
    summarized_text = asyncio.run(summarize_text(original_page_content))
    #logger.debug(f"Summarized text: {summarized_text} for document: {doc.metadata.get('title', 'No title')}")
    if not summarized_text:
        logger.warning(f"No summary generated for document: {doc.metadata.get('title', 'No title')}")
        return None
    else:
        summary_doc = Document(page_content=summarized_text)

        # add metadata, mainly from the first document
        summary_doc = enrich_document_from_parent_document(summary_doc, doc, "summary")

        # add extended page content
        if include_summary_in_search_results:
            # index = serach result = summarized text:
            pass
        else:
            # index = summarized text, search result = original page content:
            summary_doc.metadata["source"] = doc.metadata.get("source", None)
            summary_doc.metadata["page_content"] = original_page_content
            summary_doc.metadata["part"] = doc.metadata.get("part", None)
            summary_doc.metadata["part_index"] = doc.metadata.get("part_index", None)
            summary_doc.metadata["extended_page_content"] = doc.metadata.get("extended_page_content", None)
            summary_doc.metadata["extended_size"] = doc.metadata.get("extended_size", None)
            summary_doc.metadata["extended_sha256"] = doc.metadata.get("extended_sha256", None)

        return summary_doc



#
# Metadata handling
#
def enrich_document_from_parent_document(
        document: Document,
        parent_document: Document,
        part_type: str | None
        ) -> Document:
    """
    Copy metadata from parent document to the document.
    Optionally extend the source URL / anker.
    """

    # Calculate some metadata
    document.metadata["sha256"] = sha256sum_str(document.page_content)
    document.metadata["size"] = len(document.page_content)

    # Construct "part" metadata
    parent_document_part = parent_document.metadata.get("part", "")
    document.metadata["part"] = f"{parent_document_part}/{part_type}"

    # Copy metadata
    document.metadata["part_index"] = parent_document.metadata.get("part_index", 0)
    document.metadata["index_build_id"] = parent_document.metadata["index_build_id"]
    document.metadata["plob_id"] = parent_document.metadata["plob_id"]
    document.metadata["title"] = parent_document.metadata["title"] 
    document.metadata["source"] = parent_document.metadata["source"]

    # Optionally extend the anker metadata
    if part_type:
        if "anker" in parent_document.metadata:
            document.metadata["anker"] = parent_document.metadata['anker']
            # extend the anker
            #separator = "-"
            #document.metadata["anker"] = f"{parent_document.metadata['anker']}{separator}{part_type}"
        else:
            # set the anker
            document.metadata["anker"] = part_type
    else:
        # no anker to add, just copy the existing one
        document.metadata["anker"] = parent_document.metadata.get("anker", "")


    return document


#def add_or_extend_url_anker(url_with_optional_anker: str, anker_to_add: str, separator: str = "-") -> str:
#    """
#    Add or extend an anker of a URL.
#
#    If the anker already exists, it will be extended with separator + anker_to_add.
#    """
#    if not url_with_optional_anker:
#        return f"#{anker_to_add}"
#
#    if "#" in url_with_optional_anker:
#        return f"{url_with_optional_anker}{separator}{anker_to_add}"
#    else:
#        return f"{url_with_optional_anker}#{anker_to_add}"
 