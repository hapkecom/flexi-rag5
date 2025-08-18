from pydantic import BaseModel
from typing import TYPE_CHECKING
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    cast,
)
from langchain_core.documents import Document
from langchain_core.load.serializable import Serializable


# Define a tupel (plob_id, documents) as type
#PlobDocuments = Tuple[str, List[Document]]

class PlobDocuments(Serializable):
    plob_id: str
    documents: List[Document]
