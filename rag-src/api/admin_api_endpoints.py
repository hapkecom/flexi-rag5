# administrative endpoints for the API
from fastapi import APIRouter, HTTPException

from index_builder_and_retrieval_search_service.build_index import get_all_doc_parts_from_sqldb, get_all_docs_from_sqldb, get_all_parts_from_sqldb

router = APIRouter()


@router.get("/admin/documents")
async def get_documents():
    """"
    Retrive all documents from database
    """
    try:
        documents = get_all_docs_from_sqldb()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/parts")
async def get_parts():
    """"
    Retrive all parts from database
    """
    try:
        parts = get_all_parts_from_sqldb()
        return {"parts": parts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/doc-parts")
async def get_doc_parts():
    """"
    Retrive all doc-parts from database
    """
    try:
        doc_parts = get_all_doc_parts_from_sqldb()
        return {"doc_parts": doc_parts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
