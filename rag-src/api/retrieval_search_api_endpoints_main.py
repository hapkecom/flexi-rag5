from fastapi import Query
from pydantic import BaseModel
from typing import List, Optional
from index_builder_and_retrieval_search_service.search_index import search
from langchain_core.documents import Document

import logging
from fastapi import APIRouter, HTTPException
import json
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)

router = APIRouter()

class SearchResult(BaseModel):
    title: Optional[str] = None
    content: str
    url: Optional[str] = None

# SearxNG API response
class SearxNGResponse(BaseModel):
    query: str
    engines: List[str]
    results: List[SearchResult]
    answers: List[str]
    suggestions: List[str]
    infoboxes: List[str]
    unresponsive_engines: List[str]
    timing: dict
      # "total": 0.345,
      # "processing": 0.123,
      # "network": 0.222


@router.get("/search", response_model=SearxNGResponse)
async def search_endpoint(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(10, description="Maximum number of results to return"),
    engines: Optional[str] = Query(None, description="Search engines to use")
) -> SearxNGResponse:
    """
    Search endpoint implementing the SearxNG API protocol.
    """

    logger.info(f"API Received query: {q}")

    search_results = []

    # Skip searches for images or videos
    if engines:
        e = engines.lower()
        if "image" in e or "video" in e:
            logger.info(f"API Search query contains 'image' or 'video', skipping search. Engines: {engines}")

    else:
        # Perform the search using the imported search function
        content_docs: List[Document] = await search(q)

        # Check if response is None
        if content_docs:
            logger.info(f"API Response: found {len(content_docs)} documents for query '{q}':")
            for idx, doc in enumerate(content_docs):
                logger.info(f"Document {idx+1}: {str_limit(str(doc), 10000)}")
        else:
            logger.info(f"API Response: no contents (documents) found for query '{q}'")

        # Format results
        for content_doc in content_docs:
            # convert to SearchResult
            searchResult = SearchResult(
                title=content_doc.metadata.get("title", None),
                content=content_doc.page_content,
                url=content_doc.metadata.get("source", None),
            )
            search_results.append(searchResult)

    # Create SearxNGResponse object
    searxng_response = SearxNGResponse(
        query=q,
        engines=["rag"],
        results=search_results,
        answers=[],
        suggestions=[],
        infoboxes=[],
        unresponsive_engines=[],
        timing={
            "total": 0.0,
            "processing": 0.0,
            "network": 0.0
        }
    )

    logger.debug("-------------------")
    logger.debug(f"SearxNG Response - {len(search_results)} results: {str_limit(str(searxng_response), 100)}")
    logger.debug("-------------------")

    return searxng_response
