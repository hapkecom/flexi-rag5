### Question Re-writer

import logging
from async_lru import alru_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import common.service.config as config
from factory.llm_factory import get_rewrite_question_chat_llm
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)


@alru_cache(ttl=config.responseCacheTtlSeconds, maxsize=config.maxCachedQuestions)
async def rewrite_question_for_vectorsearch_retrieval(question: str) -> str:
    """
    Rewrite a question for vectorstore retrieval.
    """

    # LLM
    llm = get_rewrite_question_chat_llm()

    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the input and try to identify the underlying semantic intent / meaning. \n
         Answer with just the improved question."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n\n Formulate an improved question. Answer with just this **improved question**:\n\n",
            ),
        ]
    )

    # Action
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    updated_question = question_rewriter.invoke({"question": question})

    # Result
    logger.info(f"Updated question: '{question}' -> '{str_limit(updated_question, 150)}'")
    return updated_question


@alru_cache(ttl=config.responseCacheTtlSeconds, maxsize=config.maxCachedQuestions)
async def rewrite_question_for_keywordsearch_retrieval(question: str) -> str:
    """
    Rewrite a question for keywordsearch retrieval.
    """

    # LLM
    llm = get_rewrite_question_chat_llm()

    # Prompt
    system = """You a question analyze that identifies the most relevant single keyword for optimal search. \n 
         Look at the input and try to identify the underlying semantic intent / meaning. \n
         Answer with just this **single word**."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n\n Identifies the most relevant single keyword. Answer with just this **single word**:\n\n",
            ),
        ]
    )

    # Action
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    updated_question = question_rewriter.invoke({"question": question})

    # Result
    logger.info(f"Updated question: '{question}' -> '{str_limit(updated_question, 150)}'")
    return updated_question



@alru_cache(ttl=config.responseCacheTtlSeconds, maxsize=config.maxCachedQuestions)
async def create_hypothetical_answer_for_hyde(question: str) -> str:
    """
    Generate a hypothetical answer using an LLM-based template,
    later calculate its embedding, and use it to find more relevant documents

    HyDE (Hypothetical Document Embeddings): 
    - https://bdtechtalks.com/2024/10/06/advanced-rag-retrieval/
    - https://mikulskibartosz.name/advanced-rag-techniques-explained
    """

    # LLM
    llm = get_rewrite_question_chat_llm()

    # Prompt
    system = """You are an expert tech and biz writer tuned to produce standalone,
                ~100 - 200 token excerpts For each user query, generate exactly one
                “hypothetical document” that:\n
                - Uses a neutral, formal tone\n
                - Targets ~150 tokens total (about 2-3 paragraphs)\n
                - Contains no internal commentary or meta-instructions—only the *only* finished excerpt\n
                After these instructions, the model will receive a user prompt of the form:\n
                \n
                Query: \n\n“<USER QUERY>”\n\n Respond with the Hypothetical Document of about 150 tokens:\n
                """
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Query: \n\n {question} \n\n Respond with the Hypothetical Document of about 150 tokens:\n\n",
            ),
        ]
    )

    # Action
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    hypothetical_answer = question_rewriter.invoke({"question": question})

    # Result
    logger.info(f"Hypothetical_answer: '{question}' -> '{str_limit(hypothetical_answer, 150)}'")
    return hypothetical_answer
