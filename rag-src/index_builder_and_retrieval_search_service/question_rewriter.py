### Question Re-writer

import logging
from async_lru import alru_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import common.service.config as config
from factory.llm_factory import get_rewrite_question_chat_llm

logger = logging.getLogger(__name__)


@alru_cache(maxsize=config.maxCachedQuestions)
async def rewrite_question_for_vectorsearch_retrieval(question: str) -> str:
    """
    Rewrite a question for vectorstore retrieval.
    """

    # LLM
    llm = get_rewrite_question_chat_llm()

    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    # Action
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    updated_question = question_rewriter.invoke({"question": question})

    # Result
    logger.info(f"Updated question: '{question}' -> '{updated_question}'")
    return updated_question


@alru_cache(maxsize=config.maxCachedQuestions)
async def rewrite_question_for_keywordsearch_retrieval(question: str) -> str:
    """
    Rewrite a question for keywordsearch retrieval.
    """

    # LLM
    llm = get_rewrite_question_chat_llm()

    # Prompt
    system = """You a question analyze that identifies the most relevant single keyword for optimal search. \n 
         Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Identifies the most relevant single keyword.",
            ),
        ]
    )

    # Action
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    updated_question = question_rewriter.invoke({"question": question})

    # Result
    logger.info(f"Updated question: '{question}' -> '{updated_question}'")
    return updated_question



@alru_cache(maxsize=config.maxCachedQuestions)
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
                ~200 - 400 token excerpts For each user query, generate exactly one
                “hypothetical document” that:\n
                - Uses a neutral, formal tone\n
                - Targets ~300 tokens total (about 3-4 paragraphs)\n
                - Contains no internal commentary or meta-instructions—only the finished excerpt\n
                After these instructions, the model will receive a user prompt of the form:\n
                \n
                Query: “<USER QUERY>”\n
                Hypothetical Document:\n
                \n
                and should emit *only* the completed excerpt."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Query: \n\n{question}\n\n Hypothetical Document:\n\n",
            ),
        ]
    )

    # Action
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    hypothetical_answer = question_rewriter.invoke({"question": question})

    # Result
    logger.info(f"Hypothetical_answer: '{question}' -> '{hypothetical_answer}'")
    return hypothetical_answer
