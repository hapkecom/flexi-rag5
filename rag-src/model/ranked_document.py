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


# define a tupel (rank, document) as type
RankedDocument = Tuple[int, Document]
