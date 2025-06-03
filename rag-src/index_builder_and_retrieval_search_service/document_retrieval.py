### Retrieval of (Graded) Documents

import logging
from typing import List
from langchain_core.documents import Document

from factory.vectorstore_factory import get_vectorstore
from .document_retrieval_grader import grade_documents_for_question
from .question_rewriter import rewrite_question_for_vectorsearch_retrieval, rewrite_question_for_keywordsearch_retrieval

from async_lru import alru_cache
import common.service.config as config
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)


@alru_cache(maxsize=config.maxCachedQuestions)
async def get_relevant_documents_tuned(question: str) -> List[Document]:
    """Get relevant documents for a given question.
       Enrich with a tuned question
    """

    # get the relevant documents
    relevant_docs: List[Document] = await get_relevant_documents(question)
    logger.info(f"AAAAAAAA Found {str(len(relevant_docs))} relevant docs without tuned question")

    # do I need to enrich further to fine more documents?
    if relevant_docs is None or len(relevant_docs) == 0:
        # yes
        logger.info("BBBBBBBB More relevant documents needed: rewrite question for vectorsearch retrieval")

        # improve the question for vectorsearch retrieval
        tuned_question_str: str = await rewrite_question_for_vectorsearch_retrieval(question)

        # get the relevant documents (again)
        relevant_docs = await get_relevant_documents(tuned_question_str)
        logger.info(f"BBBBBBBB Found {str(len(relevant_docs))} relevant docs with 1x tuned question")

        # do I need to enrich further to fine more documents?
        if relevant_docs is None or len(relevant_docs) == 0:
            # yes
            logger.info("CCCCCCCC More relevant documents needed: rewrite question for keywordsearch retrieval")

            # improve the question for keywordsearch retrieval
            tuned2_question_str: str = await rewrite_question_for_keywordsearch_retrieval(tuned_question_str)

            # get the relevant documents (again)
            relevant_docs = await get_relevant_documents(tuned2_question_str)
            logger.info(f"CCCCCCCC Found {str(len(relevant_docs))} relevant docs with 2x tuned question")

    return relevant_docs


@alru_cache(maxsize=config.maxCachedQuestions)
async def get_relevant_documents(question: str) -> List[Document]:
    """Get relevant documents for a given question."""

    # Retrive documents
    vectorStore = get_vectorstore()
    docs = vectorStore.similarity_search(question, k=4)
    logger.debug(f"XXXXX Found {str(len(docs))} docs in vectorstore (un-graded candidates) for question: 'question'")
    for doc in docs:
        logger.debug(f"doc={doc.metadata} content='{str_limit(doc.page_content, 1000)}'")

    # Grade the documents
    relevant_docs = await grade_documents_for_question(question, docs)
   
    # Result
    logger.info(f"found {str(len(relevant_docs))} relevant docs out of {str(len(docs))} candidates")
    return relevant_docs
