### Retrieval/Document Grader

import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from factory.llm_factory import get_document_grader_chat_llm
#from rag_index_service.build_index import get_vectorstore, get_vectorstore_retriever, vectorStoreRetriever
from common.utils.string_util import str_limit
from model.ranked_document import RankedDocument

logger = logging.getLogger(__name__)


#
# Binray grading of documents
#
async def filter_documents_based_on_binary_grade_for_question(question: str, documents: List[Document]) -> List[Document]:
    """
    Grade documents for a given question with an LLM (binary score).

    Filter based on the binary score of relevance.
    """

    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    llm = get_document_grader_chat_llm()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n
        Just return 'yes' or 'no' as the answer. \n"""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    # Combine the prompt and the LLM
    retrieval_grader = grade_prompt | structured_llm_grader

    # Iterate over the documents
    # TODO: make this parallel/async with ainvoke
    relevant_docs: List[Document] = []
    for doc in documents:
        doc_txt = doc.page_content
        relevance_binary_score = retrieval_grader.invoke({"question": question, "document": doc_txt})
        logger.debug(f"relevance_binary_score={relevance_binary_score} for doc={str_limit(doc_txt, 1000)}")
        if (relevance_binary_score.binary_score == "yes"):
            relevant_docs.append(doc)

    # Result
    logger.info(f"Found {str(len(relevant_docs))} relevant docs out of {str(len(documents))} candidates")
    return relevant_docs


#
# Numeric grading/ranking of documents
#
async def filter_and_sort_documents_by_numeric_relevance_score_for_question(
        question: str,
        documents: List[Document]
        ) -> List[Document]:
    """
    Calculate a numeric score of relevance for each document.
    Filter out all completely irrelevant documents.
    Sort the documents by their relevance score: most relevant first, less relevant last.

    This is a more advanced version of the document grader.
    """
    # Action
    result_ranked_docs: List[RankedDocument] = await _filter_and_sort_documents_by_numeric_relevance_score_for_question(question, documents)

    # Convert the list of tuples back to a list of Documents
    result_docs: List[Document] = [doc for _, doc in result_ranked_docs]
    return result_docs


async def _filter_and_sort_documents_by_numeric_relevance_score_for_question(
        question: str,
        documents: List[Document]
        ) -> List[RankedDocument]:
    """
    Calculate a numeric score of relevance for each document.
    Filter out all completely irrelevant documents.
    Sort the documents by their relevance score: most relevant first, less relevant last.

    This is a more advanced version of the document grader.
    """

    minimum_relevance_score = 10

    # Data model
    class DocumentRelevanceScore(BaseModel):
        """Numeric score for relevance check on retrieved document."""

        numeric_score: int = Field(
            description="How relevant is a documents to the question, 0 means not relevant at all, 100 means very relevant"
        )

    # LLM with function call
    llm = get_document_grader_chat_llm()
    structured_llm_grader = llm.with_structured_output(DocumentRelevanceScore)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        The goal is to filter out erroneous retrievals and to sort documents by relevance. \n
        Give a integer between 0 and 100 as score to indicate whether the document is relevant to the question.\n
        0 means not relevant at all, 100 means very relevant.\n
        Just return the number as the answer. \n"""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    # Combine the prompt and the LLM
    retrieval_grader = grade_prompt | structured_llm_grader

    # Iterate over the documents
    # TODO: make this parallel/async with ainvoke
    scored_docs: List[RankedDocument] = []
    for i, doc in enumerate(documents):
        doc_txt = doc.page_content
        try:
            relevance_score = retrieval_grader.invoke({"question": question, "document": doc_txt})
            logger.debug(f"relevance_core={relevance_score} for #{i+1} doc={str_limit(doc_txt, 1000)}")
            if (relevance_score.numeric_score >= minimum_relevance_score):
                scored_docs.append((relevance_score.numeric_score, doc))
        except Exception as e:
            logger.warning(f"Error grading #{i+1} doc={str_limit(doc_txt)} - use minimum relevance score as fallback: {e}")
            scored_docs.append((minimum_relevance_score, doc))

    # Sort the documents by relevance score, most relevant first
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Result
    logger.debug(f"Found {str(len(scored_docs))} relevant ranked docs out of {str(len(documents))} candidates")
    return scored_docs
