from typing import Iterator

from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders import BlobLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from model.plob import Plob
from common.plob_creator import create_plob_with_metadata_of_blob

import logging


logger = logging.getLogger(__name__)



class BlobParserDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, blobLoader: BlobLoader, blobParser: BaseBlobParser) -> None:
        """Initialize the loader with a blobLoader and a blobParser.

        The blobLoader is used to load the files that are then parsed by the blobParser.

        Args:
            blobLoader: the blobLoader to load the files that are then parsed
            blobParser: the blobParser to parse the loaded files
        """
        self.blobLoader = blobLoader
        self.blobParser = blobParser

    def __str__(self) -> str:
        return f"BlobParserDocumentLoader(blobLoader: {self.blobLoader}, blobParser: {self.blobParser})"


    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        return self.lazy_load_documents()


    def lazy_load_documents(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """
        A lazy loader that loads blobs and parses them into documents.
        Blob by blob.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        logger.info(f"Downloading files with blobLoader: {self.blobLoader} and parsing them with blobParser: {self.blobParser}")

        # Blob by blob
        for blob in self.blobLoader.yield_blobs():
            # extract text from downloaded file
            logger.info(f"Blob to parse: {blob}")

            try:
                # parse
                documents = list(self.blobParser.lazy_parse(blob))

                # result
                logger.info(f"Extracted documents yielded now: {documents}")
                yield from documents

            except Exception as e:
                logger.warning(f"Error while parsing blob {blob}: {e} - continue with next")
                continue

    def lazy_load_plobs(self) -> Iterator[Plob]:  # <-- Does not take any arguments
        """
        A lazy loader that loads blobs and parses each blob into one plob,
        where each plob consists of one or multiple documents.
        Blob by blob.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        logger.info(f"Downloading files with blobLoader: {self.blobLoader} and parsing them with blobParser: {self.blobParser}")

        # Blob by blob
        for blob in self.blobLoader.yield_blobs():
            # Extract text from downloaded file
            logger.debug(f"Blob to parse: {blob}")

            try:
                # Check if blobParser contains function "lazy_parse2media"
                if hasattr(self.blobParser, "lazy_parse2plob"):
                    # Parse directly to plob
                    plob = list([self.blobParser.lazy_parse2plob(blob)])[0]

                    # Result
                    logger.info(f"Extracted plob directly - yielded now: {plob}")
                    yield plob
                else:
                    # Parse regularly - to documents
                    documents = list(self.blobParser.lazy_parse(blob))

                    # Create plob containing the documents
                    plob = create_plob_with_metadata_of_blob(blob)
                    plob.documents = list(documents)

                    # Result
                    logger.info(f"Extracted documents and created plob - yielded now: {plob} with {len(plob.documents)} documents")
                    yield plob
            except Exception as e:
                logger.warning(f"Error while parsing blob {blob}: {e} - continue with next")
                continue