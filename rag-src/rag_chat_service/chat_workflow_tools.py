"""
Definition of the logic/the functions 
used by the chat workflow.
"""

from functools import lru_cache
from typing import Dict, List, Optional

from async_lru import alru_cache
from langchain.schema import Document

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage
from index_builder_and_retrieval_search_service import document_retrieval
from common.service.configloader import deep_get, settings
from common.utils.document_util import get_document_source

import common.service.config as config

import logging
    
from common.utils.string_util import str_limit


logger = logging.getLogger(__name__)


#
# Types
#
class Question:
    """
    Represents the question that is relevant for this workflow.
    Usually the last question asked by the user.

    Attributes:
        message_index: The index of the message in the message list
        original_content: The content of the question, not yet enriched by document snippets
    """

    message_index: int
    original_content: str
    enriched_content: Optional[str]

    def __init__(self, message_index: int, original_content: str, enriched_content: Optional[str]):
        self.message_index = message_index
        self.original_content = original_content
        self.enriched_content = enriched_content


#
# Initial setup
#

default_llm_with_streaming = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
default_llm_without_streaming = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=False)


#
# Functions used by the chat workflow
#

async def enrich_questions_with_retrieved_documents(
        messages: List[AnyMessage],
        config: RunnableConfig
    ) -> List[AnyMessage]:
    """
    Enrich questions in messages with retrieved documents

    Args:
        messages (List[AnyMessage]): The list of messages
        config (RunnableConfig): The current runnable configuration

    Returns:
        List[AnyMessage]: The updated list of messages
    """
    # identify questions
    questions = identify_questions(messages)
    # enrich the questions in messages
    for question in questions:
        # single question
        enriched_question = await enrich_question_with_retrieved_documents(question, config)
        # set in messages
        messages[enriched_question.message_index]["content"] = enriched_question.enriched_content
    return messages


async def enrich_question_with_retrieved_documents(question: Question, config: RunnableConfig) -> Question:
    # anything to do?
    if not question.enriched_content:
        # yes
        question.enriched_content = await enrich_question_str_with_retrieved_documents(
            question.original_content) #, config)

    return question


@alru_cache(maxsize=config.maxCachedQuestions)
async def enrich_question_str_with_retrieved_documents(
    question_str: str,
    #config: RunnableConfig
) -> str:
    """
    Retrieve documents relevant to the question
    and enrich the question with the retrieved documents

    Args:
        question (str): The question
        config (RunnableConfig): The current runnable configuration

    Returns:
        str: The enriched question
    """

    logger.info("---ENRICH QUESTION WITH RETRIEVED DOCUMENTS---")

    # get the relevant documents
    relevant_docs: List[Document] = await document_retrieval.find_relevant_documents_tuned(question_str)

    # enrich the question with the retrieved documents
    enriched_question = attach_documents_to_question_str(question_str, relevant_docs)

    logger.info(f"enriched_question: {str_limit(enriched_question, 1024)}")

    return enriched_question


def attach_documents_to_question_str(
        question_str: str,
        documents_to_attach: List[Document]) -> str:
    """
    Attach documents to the messages (question)
    """
    result = question_str
    for i, doc in enumerate(documents_to_attach):
        # get parts of the document
        page_content = doc.page_content
        source = get_document_source(doc)

        # merge the parts
        result += f"\n\nDocument {i+1}\n{page_content}"
        if source:
            result += f"\nReference URL: {source}"
    
    return result


def vectorsearch_document_retrieval():
    # TODO: implement
    return list()

#def fulltextsearch_document_retrieval():
#    # TODO: implement
#    return list()

def grade_documents_for_question():
    # TODO: implement
    return list()

def transform_retrieval_question_for_vectorsearch_document_retrieval():
    # TODO: implement
    return list()

#def transform_retrieval_question_for_fulltextsearch_document_retrieval():
#    # TODO: implement
#    return list()


def identify_questions(messages: List[AnyMessage]) -> List[Question]:
    """
    Identify questions in the messages

    Args:
        messages (List[AnyMessage]): The list of messages

    Returns:
        List[Question]: List of questions
    """   


    questions = list()
    enrich_all_user_messages_with_retrieved_documents = deep_get(
        settings, "config.rag_response.enrich_all_user_messages_with_retrieved_documents")

    # Pick one or all user messages and make them questions
    for i, message in enumerate(messages):
        try:
            if message["role"] == "user":
                question = Question(
                    message_index=i,
                    original_content=message["content"],
                    enriched_content=None)
                questions.append(question)

                # Is one question is enough?
                if not enrich_all_user_messages_with_retrieved_documents:
                    break
        except TypeError:
            # Attribute access did'n work e.g. for SystemMessage:  if message["role"] == "user"
            pass
        except AttributeError:
            # Attribute access did'n work e.g. for SystemMessage:  if message.get("role") == "user"
            pass

    # result
    questions_str = "[" +(",".join(str(q) for q in questions)) + "]"
    logger.debug(f"Identified {len(questions)} questions in messages: {questions_str}")
    return questions
