### Retrieval/Document Grader

import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document

from factory.llm_factory import get_document_summarizer_chat_llm
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)


async def summarize_text(text: str) -> str | None:
    """
    Summarize a text with LLM.
    """

    # Data model
    class TextSummary(BaseModel):
        """Summary of a text."""

        summary: str = Field(
            description="The summary of the text.",
        )

    # LLM with function call
    llm = get_document_summarizer_chat_llm()
    structured_llm_summarizer = llm.with_structured_output(TextSummary)

    # Prompt
    system = """You are a helpful assistant for text summarization. \n"""
    user = """Please summarize the following text chunk in **2–3 sentences**,
        writing the summary **in the same language as the original text**.
        Return **only** a JSON object with a single field "summary"`\n
        \n
        Do not include any additional keys or commentary.\n
        \n
        Text:\n
        {TEXT}
        """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", user),
        ]
    )

    # Combine the prompt and the LLM
    summarizer = prompt | structured_llm_summarizer

    # Iterate over the documents
    textSummary = summarizer.invoke({"TEXT": text})
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"textSummary={textSummary}    for text={str_limit(text, 1000)}")

    # Result
    if textSummary.summary:
        return textSummary.summary
    else:
        logger.warning("No summary generated.")
        #return "No summary available."
        #return text[:100] + "..." if len(text) > 100 else text
        return None
