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
from langchain_core.documents.base import BaseMedia

# 
# Plob = Parsed Large OBject
#      = basic content unit that was parsed from a Blob (=like a file)
#        that contains of one or multiple (LangChain) Document(s)
#
# Usually 1 Blob = 1 Plob = 1-many Document(s)
#
class Plob(BaseMedia):
    id: str
    url: str
    media_type: str
    file_path: str
    file_size: int
    file_sha256: str
    file_last_modified: str  # UTC

    # documents of the plob:
    documents: List[Document] = None

    # def __init__(
    #     self,
    #     id: str,
    #     url: str,
    #     media_type: str,
    #     file_path: str,
    #     file_size: int,
    #     file_sha256: str,
    #     file_last_modified: str,
    #     documents: Optional[List[Document]] = None,
    # ):
    #     super().__init__()
    #     self.id = id
    #     self.url = url
    #     self.media_type = media_type
    #     self.file_path = file_path
    #     self.file_size = file_size
    #     self.file_sha256 = file_sha256
    #     self.file_last_modified = file_last_modified
    #     self.documents = documents or []

    def __repr__(self) -> str:
        """Define the plob representation."""
        str_repr = f"Plob id={self.id}"
        if self.url:
            str_repr += f" url='{self.url}'"
        if self.file_size:
            str_repr += f" size={self.file_size}"
        if self.file_sha256:
            str_repr += f" sha256='{self.file_sha256}'"
        return str_repr

    def __str__(self) -> str:
        return self.__repr__()