from typing import Callable, List, Optional

import logging
from typing import List
from langchain_core.documents import Document
from .document_retrieval import find_relevant_documents_tuned

import common.service.config as config

logger = logging.getLogger(__name__)


async def search(question: str, max_results: Optional[int]) -> List[Document]:
    """Get relevant documents for a given question.
       Enrich with a tuned question
    """
    return await find_relevant_documents_tuned(question, max_results)


if __name__ == "__main__":
    # Example usage
    search("What happened at Interleaf?")