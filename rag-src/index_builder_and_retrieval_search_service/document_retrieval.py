### Retrieval of (Graded) Documents

import logging
from typing import List
from langchain_core.documents import Document

from factory.vectorstore_factory import get_vectorstore
from .document_retrieval_grader import filter_documents_based_on_binary_grade_for_question, filter_and_sort_documents_by_numeric_relevance_score_for_question
from .question_rewriter import rewrite_question_for_vectorsearch_retrieval, rewrite_question_for_keywordsearch_retrieval, create_hypothetical_answer_for_hyde

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
async def find_relevant_documents_tuned(question: str) -> List[Document]:
    """Get relevant documents for a given question.
       Enrich with a tuned question
    """



    # Get the relevant documents
    retrieved_docs: List[Document] = await find_documents(question)
    logger.info(f"AAAAAAAA Found {str(len(retrieved_docs))} docs without tuned question")

    # Do I need to enrich further to fine more documents?
    if enable_rewrite_question_for_vectorsearch_retrieval: # and (retrieved_docs is None or len(retrieved_docs) == 0):
        # Yes
        logger.info("VVVVVVVV Rewrite question for vectorsearch retrieval")

        # Improve the question for vectorsearch retrieval
        tuned_question_str: str = await rewrite_question_for_vectorsearch_retrieval(question)

        # Get the relevant documents (again)
        further_retrieved_docs = await find_documents(tuned_question_str)
        logger.info(f"Found {str(len(further_retrieved_docs))} further docs with 1x tuned question")
        retrieved_docs.extend(further_retrieved_docs)

    # Do I need to enrich further to fine more documents?
    if enable_rewrite_question_for_keywordsearch_retrieval: # and (retrieved_docs is None or len(retrieved_docs) == 0):
        # Yes
        logger.info("KKKKKKKK Rewrite question for keywordsearch retrieval")

        # Improve the question for keywordsearch retrieval
        tuned2_question_str: str = await rewrite_question_for_keywordsearch_retrieval(question)

        # Get the relevant documents (again)
        further_retrieved_docs = await find_documents(tuned2_question_str)
        logger.info(f"Found {str(len(further_retrieved_docs))} further docs with 2x tuned question")
        retrieved_docs.extend(further_retrieved_docs)

    # Do I need to filter the documents?
    if enable_hyde_for_vectorsearch_retrieval:
        # Yes - use HyDE (Hypothetical Document Embeddings):
        #
        # Generate a hypothetical answer using an LLM-based template,
        # calculate its embedding, and use it to find more relevant documents
        # - https://bdtechtalks.com/2024/10/06/advanced-rag-retrieval/
        # - https://mikulskibartosz.name/advanced-rag-techniques-explained
        logger.info("HHHHHHHH - Use HyDE (Hypothetical Document Embeddings)")

        # Improve the question for keywordsearch retrieval
        hypothetical_answer: str = await create_hypothetical_answer_for_hyde(question)
        logger.debug(f"HyDE hypothetical answer: {hypothetical_answer}")

        # Get the relevant documents (again)
        further_retrieved_docs = await find_documents(question, hypothetical_answer)
        logger.info(f"Found {str(len(further_retrieved_docs))} further docs with HyDE")
        retrieved_docs.extend(further_retrieved_docs)

    # Un-lazy
    retrieved_docs = list(retrieved_docs)

    # Do I need to score and to filter the documents?
    if enable_result_filtering:
        # Yes
        retrieved_docs = await filter_and_sort_documents_by_numeric_relevance_score_for_question(
            question, retrieved_docs)

        # Un-lazy
        retrieved_docs = list(retrieved_docs)

    # Result
    logger.info(f"Found {str(len(retrieved_docs))} relevant docs after all processing steps")
    for doc in retrieved_docs:
        logger.debug(f"Found doc={doc.metadata} content='{str_limit(doc.page_content, 1000)}'")

    return retrieved_docs


@alru_cache(maxsize=config.maxCachedQuestions)
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
 
    logger.debug(f"XXXXX Found {str(len(docs))} docs in vectorstore (un-graded candidates) for question: '{question}'")
    for doc in docs:
        logger.debug(f"Found doc={doc.metadata} content='{str_limit(doc.page_content, 1000)}'")

    # Grade the documents
    #relevant_docs = await filter_documents_based_on_binary_grade_for_question(question, docs)
    relevant_docs = docs
   
    # un-lazy
    relevant_docs = list(relevant_docs)

    # Result
    logger.info(f"found {str(len(relevant_docs))} relevant docs out of {str(len(docs))} candidates")
    return relevant_docs
