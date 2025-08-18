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


# Define a tupel (plob_id, document) as type
#PlobDocument = Tuple[str, Document]

class PlobDocument(Serializable):
    plob_id: str
    document: Document
