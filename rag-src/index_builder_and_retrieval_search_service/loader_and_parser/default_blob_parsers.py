from typing import Iterator, Mapping, Generator
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers.html import BS4HTMLParser
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from model.plob import Plob
from common.plob_creator import create_plob_with_metadata_of_blob
import logging

logger = logging.getLogger(__name__)


class DefaultBlob2DocumentsParser(BaseBlobParser):
    def __init__(self):
        # Define the parsers for the different mime types
        self.handlers: Mapping[str, BaseBlobParser] = {
            "text/html": BS4HTMLParser(),
            "text/plain": TextParser(),
            "application/pdf": PyPDFParser(),
        }
        self.fallback_parser = TextParser()


    def __str__(self) -> str:
        return f"DefaultBlob2DocumentsParser()"



    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Load documents from a blob."""
        return self._lazy_parse2documents(blob)

    def _lazy_parse2documents(self, blob: Blob) -> Iterator[Document]:
        """Load documents from a blob."""

        # Actual parsing
        result = self._lazy_parse_to_pure_documents(blob)

        # Post-processing: Copy metadata from the blob to the documents
        for document in result:
            # Iterate over the metadata of the blob
            for key, value in blob.metadata.items():
                # copy the metadata to the document, if it is not already present
                if key not in document.metadata:
                    document.metadata[key] = value

            # Result document
            yield document

    def _lazy_parse_to_pure_documents(self, blob: Blob) -> Iterator[Document]:
        """Load documents from a blob. Inspired by MimeTypeBasedParser.lazy_parse()."""
        mimetype = blob.mimetype

        if mimetype is None:
            raise ValueError(f"{blob} does not have a mimetype.")

        if mimetype in self.handlers:
            handler = self.handlers[mimetype]
            logger.debug(f"Parsing blob to documents: blob={blob}, handler={handler}")
            # Parse the blob using the appropriate handler
            yield from handler.lazy_parse(blob)
        else:
            if self.fallback_parser is not None:
                logger.debug(f"Parsing blob to documents FALLBACK: blob={blob} with fallback_parser={self.fallback_parser}")
                yield from self.fallback_parser.lazy_parse(blob)
            else:
                raise ValueError(f"Unsupported mime type: {mimetype}")



class DefaultBlob2PlobParser(DefaultBlob2DocumentsParser):
    def __init__(self):
        # Call the parent class constructor
        super().__init__()

    def __str__(self) -> str:
        return f"DefaultBlob2PlobParser()"


    def lazy_parse2plob(self, blob: Blob) -> Generator[Plob, None, None]:
        """Load plob including its documents from a blob."""
        logger.info(f"Parsing blob to plob: blob={blob}")

        # Parse regularly - to documents
        documents = self.lazy_parse(blob)

        # Create plob containing the documents
        plob = create_plob_with_metadata_of_blob(blob)
        plob.documents = list(documents)

        # Result
        yield plob
