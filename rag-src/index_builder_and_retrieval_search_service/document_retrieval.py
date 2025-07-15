### Retrieval of (Graded) Documents

import logging
from functools import cmp_to_key
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)
from langchain_core.documents import Document

from factory.vectorstore_factory import get_vectorstore
from .document_retrieval_grader import filter_documents_based_on_binary_grade_for_question, filter_and_sort_documents_by_numeric_relevance_score_for_question
from .question_rewriter import rewrite_question_for_vectorsearch_retrieval, rewrite_question_for_keywordsearch_retrieval, create_hypothetical_answer_for_hyde
from .document_summarizer import compact_and_deduplicate_text

from common.service.configloader import deep_get, settings

from async_lru import alru_cache
import common.service.config as config
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)


enable_result_filtering = True
enable_rewrite_question_for_vectorsearch_retrieval = deep_get(settings, "config.rag_response.rewrite_question_for_vectorsearch_retrieval", default_value=False)
enable_rewrite_question_for_keywordsearch_retrieval = deep_get(settings, "config.rag_response.rewrite_question_for_keywordsearch_retrieval", default_value=False)
enable_hyde_for_vectorsearch_retrieval = deep_get(settings, "config.rag_response.hyde_for_vectorsearch_retrieval", default_value=False)


@alru_cache(maxsize=config.maxCachedQuestions)
async def find_relevant_documents_tuned(question: str, max_results: int = 5) -> List[Document]:
    """Get relevant documents for a given question.
       Enrich with a tuned question
    """

    # Parameters
    max_max_result = 10
    max_results = min(max_results, max_max_result)

    # Get the relevant documents
    retrieved_docs: List[Document] = await find_documents(question, k=2*max_results)
    logger.info(f"Found {str(len(retrieved_docs))} docs without tuned question")

    # Do I need to enrich further to fine more documents?
    if enable_rewrite_question_for_vectorsearch_retrieval: # and (retrieved_docs is None or len(retrieved_docs) == 0):
        # Yes
        logger.debug("Rewrite question for vectorsearch retrieval")

        # Improve the question for vectorsearch retrieval
        tuned_question_str: str = await rewrite_question_for_vectorsearch_retrieval(question)

        # Get the relevant documents (again)
        further_retrieved_docs = await find_documents(tuned_question_str, k=max_results)
        logger.info(f"Found {str(len(further_retrieved_docs))} docs with 1x tuned question (Rewrite question for vectorsearch retrieval)")
        retrieved_docs.extend(further_retrieved_docs)

    # Do I need to enrich further to fine more documents?
    if enable_rewrite_question_for_keywordsearch_retrieval: # and (retrieved_docs is None or len(retrieved_docs) == 0):
        # Yes
        logger.debug("Rewrite question for keywordsearch retrieval")

        # Improve the question for keywordsearch retrieval
        tuned2_question_str: str = await rewrite_question_for_keywordsearch_retrieval(question)

        # Get the relevant documents (again)
        further_retrieved_docs = await find_documents(tuned2_question_str, k=max_results)
        logger.info(f"Found {str(len(further_retrieved_docs))} docs with 2x tuned question (Rewrite question for keywordsearch retrieval)")
        retrieved_docs.extend(further_retrieved_docs)

    # Do I need to filter the documents?
    if enable_hyde_for_vectorsearch_retrieval:
        # Yes - use HyDE (Hypothetical Document Embeddings):
        #
        # Generate a hypothetical answer using an LLM-based template,
        # calculate its embedding, and use it to find more relevant documents
        # - https://bdtechtalks.com/2024/10/06/advanced-rag-retrieval/
        # - https://mikulskibartosz.name/advanced-rag-techniques-explained
        logger.debug("Use HyDE (Hypothetical Document Embeddings)")

        # Improve the question for keywordsearch retrieval
        hypothetical_answer: str = await create_hypothetical_answer_for_hyde(question)

        # Get the relevant documents (again)
        further_retrieved_docs = await find_documents(question, hypothetical_answer, k=max_results)
        logger.info(f"Found {str(len(further_retrieved_docs))} further docs with HyDE (Hypothetical Document Embeddings)")
        retrieved_docs.extend(further_retrieved_docs)

    # Un-lazy
    retrieved_docs = list(retrieved_docs)

    # Remove duplicates
    len_before = len(retrieved_docs)
    retrieved_docs = remove_duplicates_from_documents(retrieved_docs)
    len_after = len(retrieved_docs)
    logger.info(f"Removed duplicates from {len_before} -> {len_after} retrieved docs")

    # Do I need to score and to filter the documents?
    if enable_result_filtering:
        # Yes
        len_before = len(retrieved_docs)
        retrieved_docs = await filter_and_sort_documents_by_numeric_relevance_score_for_question(
            question, retrieved_docs)

        # Un-lazy
        retrieved_docs = list(retrieved_docs)
 
        len_after = len(retrieved_docs)
        logger.info(f"Filtered (and sorted): from {len_before} -> {len_after} retrieved docs")

    # Merge documents form the same source / same URL (except anker)
    len_before = len(retrieved_docs)
    retrieved_docs = await merge_documents_per_plob_id(retrieved_docs)
    len_after = len(retrieved_docs)
    logger.info(f"Merged from {len_before} -> {len_after} retrieved docs")

    # Filter and sort result documents again
    if enable_result_filtering:
        # Yes
        retrieved_docs = await filter_and_sort_documents_by_numeric_relevance_score_for_question(
            question, retrieved_docs)

        # Un-lazy
        retrieved_docs = list(retrieved_docs)

    # Final results limit enforcement
    if max_results > 0:
        retrieved_docs = retrieved_docs[:max_results]

    # Result
    logger.info(f"Found {str(len(retrieved_docs))} relevant docs after all processing steps")
    for doc in retrieved_docs:
        logger.debug(f"Found doc={doc.metadata} content='{str_limit(doc.page_content, 1000)}'")

    return retrieved_docs


#
# Grouping and merging search results
#

def remove_duplicates_from_documents(documents: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on their metadata and content.
    
    Args:
        documents (List[Document]): List of documents to filter.
    
    Returns:
        List[Document]: List of documents with duplicates removed.
    """
    seen_doc_sha256 = set()
    unique_documents = []
    for doc in documents:
        doc_sha256 = doc.metadata.get('sha256', None)  # Use a unique identifier from metadata
        if doc_sha256 == None:
            logger.warning(f"Document {doc.metadata.get('title', 'No title')} of plob_id={doc.metadata.get('plob_id', None)} has no sha256 metadata, skipping duplicate check.")
            continue
        if doc_sha256 not in seen_doc_sha256:
            seen_doc_sha256.add(doc_sha256)
            unique_documents.append(doc)

    return unique_documents


async def merge_documents_per_plob_id(documents: List[Document]) -> List[Document]:
    """
    Merge documents per plob_id into a single document.
    
    Args:
        documents (List[Document]): List of documents to merge.
    
    Returns:
        List[Document]: List of merged documents, one per plob_id.
    """
    merged_documents = await _merge_documents_per_plob_id(documents)
    return list(merged_documents.values())  # Convert dict values to list


async def _merge_documents_per_plob_id(documents: List[Document]) -> Dict[str, Document]:
    """
    Merge documents per plob_id into a single document.
    
    Args:
        documents (List[Document]): List of documents to merge.
    
    Returns:
        Dict[str, Document]: Dictionary with plob_id as keys and merged Document as values.
    """
    documents_by_plob_id: Dict[str, List[Document]] = group_documents_by_plob_id(documents)
    merged_documents: Dict[str, Document] = {}
    for plob_id, docs in documents_by_plob_id.items():
        merged_doc = await merge_documents_to_single_document(docs)
        if merged_doc:
            merged_documents[plob_id] = merged_doc
        else:
            logger.warning(f"No documents to merge for plob_id={plob_id}")
    return merged_documents


async def merge_documents_to_single_document(documents: List[Document]) -> Document | None:
    """
    Merge all documents (mainly the page content) into a single document.
    
    Args:
        documents (List[Document]): List of documents to merge.
    
    Returns:
        Document | None: A single merged document or None if the list is empty.
    """
    if not documents:
        return None

    # Does any document coatins a summary?
    has_summary = any("/summary" in doc.metadata.get('part', '') for doc in documents)
    logger.info(f"Documents to merge: {len(documents)} documents, has_summary={has_summary}")

    # Sort first by part
    documents = _sort_documents_of_a_plob_by_part(documents)

    # Merge the page content
    contents = [doc.page_content for doc in documents if doc.page_content]
    merged_content = _merge_strings_with_with_overlap_detection_and_tail_recursion("", contents)

    # Rewrite merged content if it conains a summary
    if has_summary:
        len_before = len(merged_content)
        logger.debug(f"Merged_content contains a summary, rewriting it now: '{merged_content}'")
        merged_content = await compact_and_deduplicate_text(merged_content)
        len_after = len(merged_content)
        logger.debug(f"Merged_content after rewriting: '{merged_content}'")
        logger.info (f"Rewrote merged content from {len_before} -> {len_after} characters")
    else:
        logger.debug(f"Merged content (len={len(merged_content)}) does not contain a summary, skipping rewriting")

    # Copy the metadata, with first document as a base
    base_metadata = documents[0].metadata.copy()
    metadata: Dict[str, Any] = {}
    metadata["index_build_id"] = base_metadata['index_build_id']
    metadata["plob_id"] = base_metadata['plob_id']
    metadata["source"] = base_metadata["source"]
    metadata["title"] = base_metadata["title"]
    metadata["part"] = "/merged"

    # Create a new Document with the merged content and metadata
    merged_document = Document(page_content=merged_content, metadata=metadata)

    return merged_document


def _sort_documents_of_a_plob_by_part(documents: List[Document]) -> List[Document]:
    """
    Sort documents of a plob by their 'part' metadata.
    
    Args:
        documents (List[Document]): List of documents to sort.
    
    Returns:
        List[Document]: Sorted list of documents.
    """

    # Sort documents by their 'part' metadata using a custom comparison function
    documents.sort(key=cmp_to_key(_comparison_function_for_documents_by_part))
    return documents


def _comparison_function_for_documents_by_part(doc1: Document, doc2: Document) -> int:
    """
    Comparison function for sorting documents by their 'part' metadata.

    Example parts:
        /split/5
        /split/5/summary
        /split/5/summary/join
        /split/13
        /split/13/summary
        /split/14
        /split/14/summary

    Become:
        /split/5
        /split/13
        /split/14
        /split/13/summary
        /split/5/summary
        /split/14/summary
        /split/5/summary/join

    Args:
        doc1 (Document): First document to compare.
        doc2 (Document): Second document to compare.
    
    Returns:
        int: Negative if doc1 < doc2, positive if doc1 > doc2, zero if equal.
    """
    part1: str = doc1.metadata.get('part', '')
    part2: str = doc2.metadata.get('part', '')

    # Split the parts into components
    part1_components = part1.split('/')
    part2_components = part2.split('/')

    # Compare the number of components first
    if len(part1_components) < len(part2_components):
        return -1
    elif len(part1_components) > len(part2_components):
        return 1

    # Compare the components one by one
    for comp1, comp2 in zip(part1_components, part2_components):
        if comp1.isdigit() and comp2.isdigit():
            # Compare as integers if both components are digits
            if int(comp1) < int(comp2):
                return -1
            elif int(comp1) > int(comp2):
                return 1
        else:
            # Compare as strings otherwise
            if comp1 < comp2:
                return -1
            elif comp1 > comp2:
                return 1
    # If all components are equal, compare by length
    if len(part1_components) < len(part2_components):
        return -1
    elif len(part1_components) > len(part2_components):
        return 1
    else:
        return 0


def group_documents_by_plob_id(documents: List[Document]) -> Dict[str, List[Document]]:
    """
    Group documents by their plob_id.
    
    Args:
        documents (List[Document]): List of documents to group.
    
    Returns:
        Dict[str, List[Document]]: Dictionary with plob_id as keys and lists of documents as values.
    """
    grouped_documents = {}
    for doc in documents:
        plob_id = doc.metadata.get('plob_id', None)
        if plob_id not in grouped_documents:
            grouped_documents[plob_id] = []
        grouped_documents[plob_id].append(doc)
    
    return grouped_documents

#
# Pure search functions
#

# DONT CACHE BECAUSE OF CONTINOUS INDEX UPDATED:
# @alru_cache(maxsize=config.maxCachedQuestions)
async def find_documents(
    question: str,
    alternative_str_for_embedding: str | None = None,
    k: int = 5
    ) -> List[Document]:
    """Get relevant documents for a given question.
    
    
    Args:
        question (str): The question to search for.
        alternative_str_for_embedding (str | None): An alternative string to use for embedding instead of
                                                    the question. This can be useful if you want to use a
                                                    different string for the embedding process.
        k: Number of Documents to return. Defaults to 5.
    """

    # Retrive documents
    vectorStore = get_vectorstore()
    str_for_embedding = alternative_str_for_embedding if alternative_str_for_embedding else question
    docs = vectorStore.similarity_search(str_for_embedding, k)
 
    logger.debug(f"Found {str(len(docs))} docs in vectorstore (un-graded candidates) for question: '{question}'")
    for doc in docs:
        logger.debug(f"Found doc={doc.metadata} content='{str_limit(doc.page_content, 1000)}'")

    # Grade the documents
    #relevant_docs = await filter_documents_based_on_binary_grade_for_question(question, docs)
    relevant_docs = docs
   
    # un-lazy
    relevant_docs = list(relevant_docs)

    # Result
    logger.debug(f"found {str(len(relevant_docs))} relevant docs out of {str(len(docs))} candidates")
    return relevant_docs


#
# String helper functions: overlap detection and merging
#

def _merge_strings_with_with_overlap_detection_and_tail_recursion(merged: str, remaining: List[str]) -> str:
    """
    Merge strings with overlap detection and tail recursion.

    Overlap handling is used to compensate overlapping from document splitting.
    
    Args:
        strings (List[str]): List of strings to merge.
    
    Returns:
        str: Merged string with overlaps handled.
    """
    if not remaining:
        return merged

    # Take the first string from the remaining list
    current_string = remaining[0]
    remaining_strings = remaining[1:]

    # Check for overlap with the merged string
    merged = _merge_two_strings_with_with_overlap_detection(merged, current_string)
    if not remaining_strings:
        # If there are no remaining strings, return the merged result
        return merged
    else:
        # Recursively merge the remaining strings
        return _merge_strings_with_with_overlap_detection_and_tail_recursion(merged, remaining_strings)


def _merge_two_strings_with_with_overlap_detection(s1: str, s2: str) -> str:
    """
    Merge two strings with overlap detection.

    Overlap handling is used to compensate overlapping from document splitting.
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
    
    Returns:
        str: Merged string with overlaps handled.
    """

    # Check for overlap with the first string in a range of half the length of the second string to 1
    max_allowed_overlap_len = 300
    min_overlap_len = 10
    overlap_length = min(min(len(s1), len(s2)) // 2, max_allowed_overlap_len)
    for i in range(overlap_length, min_overlap_len, -1):
        if s1.endswith(s2[:i]):
            # Overlap detected, merge without overlap
            logger.info(f"Overlap detected of len: {i}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Overlap detected: '{s1[-i:]}'")
                logger.debug(f"  Overlap detected in s1: '{s1}'")
                logger.debug(f"  Overlap detected in s2: '{s2}'")

            return s1 + s2[i:]

    # No overlap, just append
    return s1 + s2
