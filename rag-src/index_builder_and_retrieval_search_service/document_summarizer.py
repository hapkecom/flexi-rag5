### Retrieval/Document Grader

import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import time

from factory.llm_factory import get_document_summarizer_chat_llm
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)


async def summarize_text(text: str) -> str | None:
    """
    Summarize a text with LLM.
    """

    # Start time (for calculation of processing time)
    start_time = time.monotonic()

    try:
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
        used_millis = (time.monotonic() - start_time) * 1000 
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"textSummary={textSummary} after {used_millis} ms for text={str_limit(text, 1000)}")

        # Result
        if textSummary.summary:
            return textSummary.summary
        else:
            logger.warning("No summary generated.")
            #return "No summary available."
            #return text[:100] + "..." if len(text) > 100 else text
            return None
    except Exception as e:
        used_millis = int((time.monotonic() - start_time) * 1000)
        text_str = str({"TEXT": text})
        logger.warning(f"Error summarizing text for text {text_str} with prompt={prompt} after {used_millis} ms: {e}")
        raise e


async def compact_and_deduplicate_text(text: str) -> str | None:
    """
    Compact text and remove duplicated content using LLM.
    """

    # Data model
    class CompactedText(BaseModel):
        """Compacted and deduplicated text."""

        compacted_text: str = Field(
            description="The compacted text with duplicates removed.",
        )

    # LLM with function call
    llm = get_document_summarizer_chat_llm()
    structured_llm_compactor = llm.with_structured_output(CompactedText)

    # Prompt
    system = """You are a helpful assistant for text compacting and deduplication. 
Your task is to remove redundant information and duplicate content while preserving all unique and important information.
You should:
1. Identify and remove duplicate sentences, phrases, or concepts
2. Consolidate repetitive information into single, clear statements
3. Maintain the original meaning and all unique facts
4. Preserve the original language and writing style
5. Keep the text coherent and well-structured
6. Do not summarize - preserve all unique content, just remove duplicates"""

    user = """Please compact the following text by removing duplicated content and redundant information.
        Keep all unique information and maintain the original language.

        Important guidelines:
        - Remove duplicate sentences, phrases, or repeated concepts
        - Consolidate similar information into single, clear statements
        - Preserve all unique facts, names, dates, and specific details
        - Maintain logical flow and coherence
        - Do not summarize - only remove duplicates
        - Write the compacted text **in the same language as the original text**

        Return **only** a JSON object with a single field "compacted_text".
        Do not include any additional keys or commentary.

        Text to compact:
        {TEXT}
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", user),
        ]
    )

    # Combine the prompt and the LLM
    compactor = prompt | structured_llm_compactor

    # Process the text
    compacted_result = compactor.invoke({"TEXT": text})
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"compacted_result={compacted_result}    for text={str_limit(text, 1000)}")

    # Result
    if compacted_result.compacted_text:
        return compacted_result.compacted_text
    else:
        logger.warning("No compacted text generated.")
        return None
