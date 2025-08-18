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
from common.utils.hash_util import sha256sum_str
from common.utils.string_util import str_limit, merge_strings_with_with_overlap_detection_and_tail_recursion, merge_two_strings_with_with_overlap_detection
from common.service.logging_tools import log_docs, doc2str
from model.plob_document import PlobDocument
from model.plob_documents import PlobDocuments

logger = logging.getLogger(__name__)


enable_intermediate_result_filtering_with_llm = deep_get(settings, "config.rag_response.intermediate_result_filtering_with_llm", default_value=True)
enable_final_result_filtering_with_llm = deep_get(settings, "config.rag_response.final_result_filtering_with_llm", default_value=False)
enable_rewrite_question_for_vectorsearch_retrieval = deep_get(settings, "config.rag_response.rewrite_question_for_vectorsearch_retrieval", default_value=False)
enable_rewrite_question_for_keywordsearch_retrieval = deep_get(settings, "config.rag_response.rewrite_question_for_keywordsearch_retrieval", default_value=False)
enable_hyde_for_vectorsearch_retrieval = deep_get(settings, "config.rag_response.hyde_for_vectorsearch_retrieval", default_value=False)
deliver_extended_content = deep_get(settings, "config.rag_response.deliver_extended_content", default_value=True)

enable_rewrite_summaries = deep_get(settings, "config.rag_response.rewrite_summaries", default_value=False)
enable_rewrite_complete_response = deep_get(settings, "config.rag_response.rewrite_complete_response", default_value=False)


default_max_search_results = deep_get(settings, "config.rag_response.default_max_search_results", default_value=10)
max_max_search_results = deep_get(settings, "config.rag_response.max_max_search_results", default_value=25)


@alru_cache(ttl=config.responseCacheTtlSeconds, maxsize=config.maxCachedQuestions)
async def find_relevant_documents_tuned(question: str, max_results: Optional[int]) -> List[Document]:
    """Get relevant documents for a given question.
       Enrich with a tuned question
    """

    # Parameters
    if max_results is None or max_results <= 0:
        max_results = default_max_search_results
    max_results = min(max_results, max_max_search_results)

    # Store result in list of list to later mix tge order
    list_of_list_of_retrieved_docs: List[List[Document]] = []

    # Get the relevant documents
    normal_retrieved_docs: List[Document] = await find_documents(question, k=2*max_results)
    logger.info(f"Found {str(len(normal_retrieved_docs))} docs with original question")
    list_of_list_of_retrieved_docs.append(normal_retrieved_docs)

    # Enrich further to fine more documents - with HyDE (Hypothetical Document Embeddings)?
    if enable_hyde_for_vectorsearch_retrieval:
        # Yes - use HyDE (Hypothetical Document Embeddings):
        #
        # Generate a hypothetical answer using an LLM-based template,
        # calculate its embedding, and use it to find more relevant documents
        # - https://bdtechtalks.com/2024/10/06/advanced-rag-retrieval/
        # - https://mikulskibartosz.name/advanced-rag-techniques-explained
        try:
            logger.info("Use HyDE (Hypothetical Document Embeddings) now ...")

            # Improve the question for keywordsearch retrieval
            hypothetical_answer: str = await create_hypothetical_answer_for_hyde(question)

            # Get the relevant documents (again)
            further_retrieved_docs = await find_documents(question, hypothetical_answer, k=max_results)
            logger.info(f"Found {str(len(further_retrieved_docs))} further docs with HyDE (Hypothetical Document Embeddings)")
            list_of_list_of_retrieved_docs.append(further_retrieved_docs)
        except Exception as e:
            # Probably LLM request failed,
            # no re-try because of performance reasons
            logger.warning(f"Error while using HyDE (Hypothetical Document Embeddings): {e}")

    # Enrich further to fine more documents - rewrite question for vectorsearch retrieval?
    if enable_rewrite_question_for_vectorsearch_retrieval: # and (retrieved_docs is None or len(retrieved_docs) == 0):
        # Yes
        try:
            logger.info("Rewrite question for vectorsearch retrieval now ...")

            # Improve the question for vectorsearch retrieval
            tuned_question_str: str = await rewrite_question_for_vectorsearch_retrieval(question)

            # Get the relevant documents (again)
            further_retrieved_docs = await find_documents(tuned_question_str, k=max_results, alpha=1.0)
            logger.info(f"Found {str(len(further_retrieved_docs))} docs with 1st tuned question (Rewrite question for vectorsearch retrieval)")
            list_of_list_of_retrieved_docs.append(further_retrieved_docs)
        except Exception as e:
            # Probably LLM request failed,
            # no re-try because of performance reasons
            logger.warning(f"Error while rewriting question for vectorsearch retrieval: {e}")

    # Enrich further to fine more documents - rewrite question for keywordsearch retrieval?
    if enable_rewrite_question_for_keywordsearch_retrieval: # and (retrieved_docs is None or len(retrieved_docs) == 0):
        # Yes
        try:
            logger.info("Rewrite question for keywordsearch retrieval now ...")

            # Improve the question for keywordsearch retrieval
            tuned2_question_str: str = await rewrite_question_for_keywordsearch_retrieval(question)

            # Get the relevant documents (again)
            further_retrieved_docs = await find_documents(tuned2_question_str, k=((1+max_results)//2), alpha=0.0)
            logger.info(f"Found {str(len(further_retrieved_docs))} docs with 2nd tuned question (Rewrite question for keywordsearch retrieval)")
            list_of_list_of_retrieved_docs.append(further_retrieved_docs)
        except Exception as e:
            # Probably LLM request failed,
            # no re-try because of performance reasons
            logger.warning(f"Error while rewriting question for keywordsearch retrieval: {e}")

    # Un-lazy
    unlazy_list_of_list_of_retrieved_docs: List[List[Document]] = []
    max_len = 0
    num_of_docs = 0
    for list_of_retrieved_docs in list_of_list_of_retrieved_docs:
        unlazy_list_of_list_of_retrieved_docs.append(list(list_of_retrieved_docs))
        l = len(list_of_retrieved_docs)
        num_of_docs += l
        if l > max_len:
            max_len = l
    # Remix docs (to ensure a good order without sorting)
    retrieved_docs: List[Document] = []
    logger.info(f"Remix docs ({len(unlazy_list_of_list_of_retrieved_docs)} lists with max_len={max_len} and {num_of_docs} docs in total)")
    for i in range(max_len):
        #logger.debug(f"    Remix i={i}/{max_len} lists")
        for list_of_retrieved_docs in unlazy_list_of_list_of_retrieved_docs:
            #logger.debug(f"        Remix docs: with list of {len(list_of_retrieved_docs)} docs")
            if i < len(list_of_retrieved_docs):
                # Add the document to the result
                #logger.debug(f"            Added doc {doc2str(list_of_retrieved_docs[i])}")
                retrieved_docs.append(list_of_retrieved_docs[i])

    # Remove duplicates
    len_before = len(retrieved_docs)
    retrieved_docs = remove_duplicates_from_documents(retrieved_docs)
    len_after = len(retrieved_docs)
    msg = f"Removed duplicates from {len_before} -> {len_after} retrieved docs"
    log_docs(logger, logging.INFO, msg, retrieved_docs)

    # Sort and filter the documents?
    docs_sources_sorted_by_relevance_score: List[str] = []
    if enable_intermediate_result_filtering_with_llm:
        # Yes: Sort and filter the documents with LLM (intermediate)
        logger.info("Filter and sort with LLM (intermediate) documents by numeric relevance score for question now ...")
        len_before = len(retrieved_docs)
        try:
            retrieved_docs = await filter_and_sort_documents_by_numeric_relevance_score_for_question(
                question, retrieved_docs)

            # Un-lazy
            retrieved_docs = list(retrieved_docs)
    
            len_after = len(retrieved_docs)
            msg = f"Filtered and sorted with LLM (intermediate): from {len_before} -> {len_after} retrieved docs"
            log_docs(logger, logging.INFO, msg, retrieved_docs)
        except Exception as e:
            # Probably LLM request(s) failed,
            # no re-try because of performance reasons
            len_after = len(retrieved_docs)
            logger.warning(f"Error while filtering and sorting documents by numeric relevance score - continue with {len_before} of {len_after} retrieved docs: {e}")
    else:
        # No: Skip filtering and sorting with LLM (intermediate),
        # keep the original retrieval order
        len_after = len(retrieved_docs)
        msg = f"Skipped filtering and sorting with LLM because of configuration (intermediate_result_filtering_with_llm=False), continue with all {len_after} retrieved docs"
        log_docs(logger, logging.INFO, msg, retrieved_docs)
    # Collect sources sorted by relevance score (for later re-use)
    for doc in retrieved_docs:
        source = doc.metadata.get('source', None)
        if source is not None and source not in docs_sources_sorted_by_relevance_score:
            docs_sources_sorted_by_relevance_score.append(source)

    # Merge documents form the same source / same URL (except anker)
    len_before = len(retrieved_docs)
    retrieved_docs = await merge_documents_per_plob_id(retrieved_docs)
    len_after = len(retrieved_docs)
    msg = f"Merged from {len_before} -> {len_after} retrieved docs"
    log_docs(logger, logging.INFO, msg, retrieved_docs)

    # Filter and sort result documents again
    if enable_final_result_filtering_with_llm:
        # Yes: Filter and sort the documents with LLM (final)
        logger.info("Final result filtering and sorting with LLM now ...")
        try:
            retrieved_docs = await filter_and_sort_documents_by_numeric_relevance_score_for_question(
                question, retrieved_docs)
        except Exception as e:
            # Probably LLM request(s) failed,
            # no re-try because of performance reasons
            logger.warning(f"Error while filtering and sorting final documents by numeric relevance score - continue with {len(retrieved_docs)} retrieved docs: {e}")

        # Un-lazy
        retrieved_docs = list(retrieved_docs)
    else:
        # No: Skip filtering and sorting with LLM (final)
        if len(docs_sources_sorted_by_relevance_score) > 0:
            # Sort by relevance score stored in docs_sources_sorted_by_relevance_score
            logger.info("Final result sorting by earlier calculated relevance score without LLM because of configuration (final_result_filtering_with_llm=False)")
            retrieved_docs = sorted(retrieved_docs, key=lambda doc: docs_sources_sorted_by_relevance_score.index(doc.metadata.get('source', '')), reverse=False)
        else:
            # No previous relevance score, sort by source
            logger.info("No final result filtering and sorting at all: neither with LLM (final_result_filtering_with_llm=False) nor by earlier calculated relevance score (non-existing)")

    # Final results limit enforcement
    if max_results > 0:
        retrieved_docs = retrieved_docs[:max_results]

    # Result
    log_docs(logger, logging.INFO, "Final retrieved docs", retrieved_docs)

    return retrieved_docs


#
# Grouping and merging search results
#

def remove_duplicates_from_documents(documents: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on their metadata and content.

    Keep the original order of the documents based on the first occurrence of each unique document.
    
    Args:
        documents (List[Document]): List of documents to filter.
    
    Returns:
        List[Document]: List of documents with duplicates removed.
    """
    seen_sha256 = set()
    unique_documents = []
    for doc in documents:
        # Use the doc.page_content as a unique identifier
        # (doc.metadata.get('sha256', None) is not always matching  because of page_content tuning))

        # Preparation
        page_content = doc.page_content
        if page_content is None:
            continue  # Skip documents with no content
        
        # Calculate and compare the SHA256 hash of the page content
        page_content_sha256 = sha256sum_str(page_content)
        if page_content_sha256 not in seen_sha256:
            seen_sha256.add(page_content_sha256)
            unique_documents.append(doc)
        else:
            logger.debug(f"Duplicate document found: {doc2str(doc)}")

    return unique_documents


async def merge_documents_per_plob_id(documents: List[Document]) -> List[Document]:
    """
    Merge documents per plob_id into a single document.

    Try to keep the order of the documents as much as possible,
    based on the first document of each plob_id.
    
    Args:
        documents (List[Document]): List of documents to merge.
    
    Returns:
        List[Document]: List of merged documents, one per plob_id.
    """
    merged_documents = await _merge_documents_per_plob_id(documents)
    return [doc.document for doc in merged_documents if doc.document is not None]


async def _merge_documents_per_plob_id(documents: List[Document]) -> List[PlobDocuments]:
    """
    Merge documents per plob_id into a single document.
    
    Args:
        documents (List[Document]): List of documents to merge.
    
    Returns:
        List[PlobDocument]: List of merged documents, one per plob_id.
    """
    documents_by_plob_id: List[PlobDocuments] = _group_documents_by_plob_id(documents)
    merged_documents: List[PlobDocument] = []
    for plob_docs in documents_by_plob_id:
        merged_doc = await merge_some_documents_of_a_plob_to_single_document(plob_docs.documents)
        plob_id = plob_docs.plob_id
        if merged_doc:
            merged_documents.append(PlobDocument(plob_id=plob_docs.plob_id, document=merged_doc))
        else:
            logger.warning(f"No documents to merge for plob_id={plob_id} - continue with next plob_id")
    return merged_documents


async def merge_some_documents_of_a_plob_to_single_document(documents: List[Document]) -> Document | None:
    """
    Merge all documents (mainly the page content) into a single document.
    
    Args:
        documents (List[Document]): List of documents to merge.
    
    Returns:
        Document | None: A single merged document or None if the list is empty.
    """
    if not documents:
        # No documents to merge
        return None
    if len(documents) == 1:
        # Only one document, return it as is
        return documents[0]

    # Sort first by part
    documents = _sort_documents_of_a_plob_by_part(documents)
    base_document = documents[0]

    # Separate documents with summaries from those without
    documents_with_summary    = [doc for doc in documents if "/summary" in     doc.metadata.get('part', '')]
    documents_without_summary = [doc for doc in documents if "/summary" not in doc.metadata.get('part', '')]
    logger.debug(f"Documents to merge: {len(documents_without_summary)} documents without summary, {len(documents_with_summary)} documents with summary")

    # Merge the page contents of documents without summary
    without_summary_contents = [doc.page_content for doc in documents_without_summary if doc.page_content]
    without_summary_merged_content = merge_strings_with_with_overlap_detection_and_tail_recursion(
        "", without_summary_contents, separator_in_case_of_simple_concatenation="\n\n...\n\n"
    )

    # Merge the page contents of documents with summary
    with_summary_contents = [doc.page_content for doc in documents_with_summary if doc.page_content]
    with_summary_merged_content = merge_strings_with_with_overlap_detection_and_tail_recursion(
        "", with_summary_contents, separator_in_case_of_simple_concatenation="\n\n...\n\n"
    )
    if len(with_summary_merged_content) > 0:
        if len(with_summary_contents) >= 2:
            # Multiple documents with summary
            if enable_rewrite_summaries:
                # Summarized the summaries
                len_before = len(with_summary_merged_content)
                logger.debug(f"Merged_content contains multiple summaries, rewriting it now: '{with_summary_merged_content}'")
                with_summary_merged_content = await compact_and_deduplicate_text(with_summary_merged_content)
                len_after = len(with_summary_merged_content)
                logger.debug(f"Merged_content contains multiple summaries, after rewriting: '{with_summary_merged_content}'")
                logger.info(f"Rewrote merged content (multiple summaries) from {len_before} -> {len_after} characters")
            else:
                # Simply use the merged content as is
                logger.debug(f"Merged_content contains multiple summaries, but not rewriting it: '{with_summary_merged_content}'")
        else:
            # Single document with summary
            logger.debug(f"Merged_content contains a single summary, not rewriting it: '{with_summary_merged_content}'")
    else:
        # No summaries at all
        logger.debug("No summaries found in the documents, skipping summary merging")

    # Merge the contents: without summary + with summary
    merged_content = ""
    logger.debug(f"enable_rewrite_complete_response={enable_rewrite_complete_response}, len(without_summary_merged_content)={len(without_summary_merged_content)}, len(with_summary_merged_content)={len(with_summary_merged_content)}")
    if not enable_rewrite_complete_response:
        # Simply merge the contents without rewriting
        if len(without_summary_merged_content) > 0 and len(with_summary_merged_content) > 0:
            # Both contents exist, merge them
            merged_content = (
                without_summary_merged_content +
                "\n\n**Summary / Zusammenfassung:**\n\n" +
                with_summary_merged_content
            )
        elif len(without_summary_merged_content) > 0:
            # Only without summary content exists
            merged_content = without_summary_merged_content
        elif len(with_summary_merged_content) > 0:
            # Only with summary content exists
            merged_content = with_summary_merged_content
        logger.info (f"Simple merged content of {len(merged_content)} characters")
    else:
        # Merge and complely rewrite the content
        merged_content = merge_two_strings_with_with_overlap_detection(
            without_summary_merged_content,
            with_summary_merged_content,
            separator_in_case_of_simple_concatenation="\n\n...\n\n"
        )
        # Rewrite merged content if it conains a summary
        len_before = len(merged_content)
        logger.debug(f"Merged_content contains a summary, rewriting it now: '{merged_content}'")
        merged_content = await compact_and_deduplicate_text(merged_content)
        len_after = len(merged_content)
        logger.debug(f"Merged_content after rewriting: '{merged_content}'")
        logger.info (f"Rewrote merged content from {len_before} -> {len_after} characters")

    # Copy the metadata, with first document as a base
    base_metadata = base_document.metadata.copy()
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

def _group_documents_by_plob_id(documents: List[Document]) -> List[PlobDocuments]:
    """
    Group documents by their plob_id.
    
    Args:
        documents (List[Document]): List of documents to group.
    
    Returns:
        List[PlobDocuments]: List of tuples, each containing a plob_id and the corresponding list of documents.
    """
    results: List[PlobDocuments] = []
    for doc in documents:
        plob_id = doc.metadata.get('plob_id', None)
        # Looking for plob_id in results
        found_blob_id = False
        for plob_docs in results:
            if plob_docs.plob_id == plob_id:
                # Found the plob_id, append the document
                plob_docs.documents.append(doc)
                found_blob_id = True
                break
        if not found_blob_id:
            # Create a new PlobDocuments instance if plob_id not found
            new_plob_docs = PlobDocuments(plob_id=plob_id, documents=[doc])
            results.append(new_plob_docs)

    return results


#
# Pure search functions
#

@alru_cache(ttl=config.responseCacheTtlSeconds, maxsize=config.maxCachedQuestions)
async def find_documents(
    question: str,
    alternative_str_for_embedding: str | None = None,
    k: int = 5,
    alpha: float = 0.75
) -> List[Document]:
    """Get relevant documents for a given question.
    
    
    Args:
        question (str): The question to search for.
        alternative_str_for_embedding (str | None): An alternative string to use for embedding instead of
                                                    the question. This can be useful if you want to use a
                                                    different string for the embedding process.
        k: Number of Documents to return. Defaults to 5.
        alpha (float): The alpha parameter for the similarity search. It controls the balance between
                       keyword search and vector search. Defaults to 0.75.
                       alpha = 0 forces using a pure keyword search method (BM25)
                       alpha = 1 forces using a pure vector search method

    """

    # Retrive documents
    vectorStore = get_vectorstore()
    str_for_embedding = alternative_str_for_embedding if alternative_str_for_embedding else question

    # similarity_search() uses Weaviate's hybrid search.
    #   https://python.langchain.com/docs/integrations/vectorstores/weaviate/#search-mechanism
    #   https://docs.weaviate.io/weaviate/api/graphql/search-operators#hybrid
    #
    # The alpha parameter controls the balance between keyword search and vector search.
    # alpha can be any number from 0 to 1, defaulting to 0.75.
    #    alpha = 0 forces using a pure keyword search method (BM25)
    #    alpha = 1 forces using a pure vector search method
    #    alpha = 0.5 weighs the BM25 and vector methods evenly
    logger.info(f"Find documents for question: '{str_limit(str_for_embedding, 150)}' (k={k}, alpha={alpha})")
    docs = vectorStore.similarity_search(str_for_embedding, k=k, alpha=alpha)
 
    # Content from metadata - if index data and search results are not the same
    consider_metadata_page_content = True
    if consider_metadata_page_content:
        # Yes, deliver extended content
        updated_docs: List[Document] = []        
        for doc in docs:
            metadata_page_content = doc.metadata.get('page_content', None)
            if metadata_page_content:
                # metadata_page_content exists, use it
                logger.debug(f"  Use metadata_page_content for: {doc2str(doc)}")
                updated_doc = doc.model_copy(deep=True)
                updated_doc.page_content = metadata_page_content
                updated_docs.append(updated_doc)
            else:
                # No metadata_page_content, use the original document
                #logger.debug(f" Don't use metadata_page_content (because it doesn't exist) for: {doc2str(doc)}")
                updated_docs.append(doc)
        docs = updated_docs

    # Extended content?
    if deliver_extended_content:
        # Yes, deliver extended content
        updated_docs: List[Document] = []
        for doc in docs:
            extended_page_content = doc.metadata.get("extended_page_content", None)
            if extended_page_content:
                # Extended content exists, use it
                logger.debug(f"  Use extended_page_content for: {doc2str(doc)}")
                updated_doc = doc.model_copy(deep=True)
                updated_doc.page_content = extended_page_content
                updated_docs.append(updated_doc)
            else:
                # No extended content, use the original document
                #logger.debug(f" Don't use extended_page_content (because it doesn't exist) for: {doc2str(doc)}")
                updated_docs.append(doc)
        docs = updated_docs

    # Grade the documents
    #relevant_docs = await filter_documents_based_on_binary_grade_for_question(question, docs)
    relevant_docs = docs
   
    # un-lazy
    relevant_docs = list(relevant_docs)

    # Result
    logger.debug(f"found {str(len(relevant_docs))} relevant docs out of {str(len(docs))} candidates")
    return relevant_docs

