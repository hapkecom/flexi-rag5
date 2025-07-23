

from typing import TYPE_CHECKING
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)
import shortuuid
from langchain_core.documents import Document
import logging
import model.plob as Plob

from common.utils.hash_util import sha256sum_str
from common.utils.string_util import str_limit, str_limit_hard_cut
import queue


#logger = logging.getLogger(__name__)


#
# Logging tools for Plobs
#
def plob2str(plob: Plob) -> str:
    """Convert a Plob to a compact string representation."""

    # Pre-check
    if plob is None:
        return "Plob=None"

    # Prepare the string representation
    id = plob.id
    id_str = f"id={str_limit(id, 40)}" if id else "id=None"

    sha256 = plob.file_sha256
    sha256_str = f"sha256={str_limit_hard_cut(sha256, 8)}" if sha256 is not None else "sha256=None"

    size = plob.file_size
    size_str = f"size={str(int(size))}" if size is not None else "size=None"

    url = plob.url
    url_str = f"url={str_limit(url, 120)}" if url else "url=None"

    media_type = plob.media_type
    media_type_str = f"media_type={str_limit(media_type, 80)}" if media_type else "media_type=None"

    # Return the string representation
    return f"Plob: {id_str}, {sha256_str}, {size_str}, {media_type_str}, {url_str}"


#
# Logging tools for documents
#

def doc2str(doc: Document) -> str:
    """Convert a Document to a compact string representation."""

    # Pre-check
    if doc is None:
        return "Document=None"

    # Prepare the string representation
    sha256 = doc.metadata.get('sha256', None)
    sha256_str = f"sha256={str_limit_hard_cut(sha256, 8)}" if sha256 is not None else "sha256=None"

    #page_content_str = _doc_attributes2str(name='page_content', sha256=doc.metadata.get('sha256', None), size=doc.metadata.get('size', None), content=doc.page_content)
    #metadata_page_content_str = _doc_attributes2str(name='metadata_page_content', sha256=doc.metadata.get('sha256', None), size=doc.metadata.get('size', None), content=doc.metadata.get('page_content', None))
    #extended_page_content_str = _doc_attributes2str(name='extended_page_content', sha256=doc.metadata.get('extended_sha256', None), size=doc.metadata.get('extended_size', None), content=doc.metadata.get('extended_page_content', None))
    page_content_str = _doc_content2str(name='page_content', content=doc.page_content)
    metadata_page_content_str = _doc_content2str(name='metadata_page_content', content=doc.metadata.get('page_content', None))
    extended_page_content_str = _doc_content2str(name='extended_page_content', content=doc.metadata.get('extended_page_content', None))

    source_str = f"source={str_limit(doc.metadata.get('source', 'None'), 120)}"
    part_str = f"part={str_limit(doc.metadata.get('part', 'None'), 80)}"
    title_str = f"title={str_limit(doc.metadata.get('title', 'None'), 120)}"

    # Return the string representation
    return f"Document: {sha256_str}, {source_str}, {part_str}, {page_content_str}, {metadata_page_content_str}, {extended_page_content_str}, {title_str}"

def _doc_content2str(name: str = "content", content: Optional[str] = None) -> str:
    """Convert the content of a document to a string representation."""
    # Pre-check
    if content is None:
        return f"{name}()=None"

    # Prepare the string representation
    sha256 = sha256sum_str(content)
    size = len(content)

    # Return the string representation
    return _doc_attributes2str(name=name, sha256=sha256, size=size, content=content)

def _doc_attributes2str(name: str = "content", sha256: Optional[str] = None, size: Optional[int] = None, content: Optional[str] = None) -> str:
    """Convert some document attributes to a string representation."""
    # Pre-check
    if sha256 is None and size is None and content is None:
        return f"{name}()=None"

    # Prepare the string representation
    if sha256 is None:
        sha256 = "None"
    if size is None:
        size = "None"
    else:
        size = str(int(size))
    if content is None:
        content = "None"
    return f"{name}(sha256={str_limit_hard_cut(sha256, 8)}/size={size})={str_limit(content, 40)}"


def log_docs(
    your_logger: logging.Logger,
    log_level: int,
    msg: str,
    docs: List[Document],
) -> None:
    """Log a list of documents."""

    if your_logger.isEnabledFor(log_level):
        your_logger.log(log_level, f"{msg}: {len(docs)} documents:", stacklevel=2)
        for doc in docs:
            your_logger.log(log_level, f"    {doc2str(doc)}", stacklevel=2)
