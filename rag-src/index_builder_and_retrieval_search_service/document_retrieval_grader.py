### Retrieval/Document Grader

import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document

from factory.llm_factory import get_document_grader_chat_llm
#from rag_index_service.build_index import get_vectorstore, get_vectorstore_retriever, vectorStoreRetriever
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)


async def grade_documents_for_question(question: str, documents: List[Document]) -> List[Document]:
    """
    Grade documents for a given question.
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
