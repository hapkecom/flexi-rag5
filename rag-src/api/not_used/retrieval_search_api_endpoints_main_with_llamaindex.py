from fastapi import Query
from pydantic import BaseModel
from typing import List, Optional
#from index_builder_and_retrieval_search_service.llamaindex_weaviate_rag import search  # Import the search function
from llama_index.core.response import Response
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

class SearchResult(BaseModel):
    title: str
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
    max_results: int = Query(10, description="Maximum number of results to return")
):
    """
    Search endpoint implementing the SearxNG API protocol.
    """
    print(f"API Received query: {q}")
    # Perform the search using the imported search function
    response: Response = None # TODO: search(q)
    #print(f"Response: {response}")

    # Check if response is None
    if response is None or not hasattr(response, 'source_nodes'):
        print(f"Response: {response}")
        return {"error": "No results found or invalid response from search function"}

    # Format results
    results = []
    for node in response.source_nodes[:max_results]:
        # convert node to SearchResult
        print(f"*** Node metadata: {node.node.metadata}")
        print(f"*** Node URL: {node.node.metadata.get('url')}")

        results.append(SearchResult(
            title=node.node.metadata.get("original_text", "No Title"),
            content=node.node.metadata.get("window", None),
            url=node.node.metadata.get("url")
        ))

    # Create SearxNGResponse object
    searxng_response = SearxNGResponse(
        query=q,
        engines=["rag"],
        results=results,
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
    print("-------------------")
    print(f"SearxNG Response: {searxng_response}")
    print("-------------------")

    return searxng_response
