import os
from typing import Any, Dict, Iterable, Iterator, Optional
#from langchain_community.document_loaders.blob_loaders.file_system import FileSystemBlobLoader
#from langchain_community.document_loaders.[blob_loaders.schema] import BlobLoader
from langchain_community.document_loaders import BlobLoader
from langchain_core.documents.base import Blob as Blob

import logging

from common.blob_creator import create_blob_from_local_file

logger = logging.getLogger(__name__)


class WgetBlobLoader(BlobLoader):
    def __init__(self,
                 url: Optional[str] = None,
                 command: Optional[str] = None,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the WgetBlobLoader with a URL or a command."""

        self.url = url
        self.command = command

        if not self.url and not self.command:
            raise ValueError("Either 'url' or 'command' must be provided to WgetBlobLoader.")

    def yield_blobs(
        self,
    ) -> Iterable[Blob]:
        """Yield blobs that match the requested pattern."""
        
        # crawl with wget and iterate over the blobs (downloaded files)
        logger.info(f"Downloading files from url/with command: {self.url} / {self.command}")
        blobs: Iterator[Blob] = []
        if self.command is None and self.url is None:
            raise ValueError("Either 'url' or 'command' must be provided to WgetBlobLoader.")
        if self.command is None:
            # use the url to crawl with wget
            logger.info(f"Using URL '{self.url}' for crawling with wget")
            blobs = WgetBlobLoader.crawl_single_url_with_wget(self.url)
        else:
            # use the command to crawl with wget
            logger.info(f"Using command for crawling with wget")
            directory_prefix = "/tmp/wget"
            blobs = WgetBlobLoader.crawl_with_single_command(self.command, directory_prefix)

        for blob in blobs:
            # extract text from downloaded blob
            logger.info(f"Downloaded file: {blob}")
            yield blob


    def __str__(self) -> str:
        return f"WgetBlobLoader(url: {self.url})"


    @staticmethod
    def crawl_single_url_with_wget(url) -> Iterator[Blob]:
        directory_prefix = "/tmp/wget"
        command = f"wget --directory-prefix {directory_prefix} --recursive -l1 --no-parent -A.html,.txt,.mp4,.pdf --limit-rate=1024k --wait=10 {url}"
        return WgetBlobLoader.crawl_with_single_command(command, directory_prefix)

    @staticmethod
    def crawl_with_single_command(command, directory_prefix) -> Iterator[Blob]:
        # crawl with command(wget) with popen
        # and read name of downloaded files from stdin/stdout (with popen)
        import subprocess
        import os
        import io

        logger.info(f"Crawl now with (wget compatible) command:    {command}")

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            shell=True
        )
        for line in io.TextIOWrapper(proc.stderr, encoding="utf-8"):  # or another encoding
            # do something with line
            # trim the line
            logger.info("    COMMAND(WGET?): " + line.strip())
            # extract <file_path> from line with "‘<file_path>’ saved"
            if "‘" in line and "’ saved" in line:
                # extract file_path from line
                file_path = line.split("‘")[1].split("’ saved")[0]
                # derive url from file_path
                if file_path.startswith(directory_prefix):
                    dir_prefix2 = directory_prefix
                    if not file_path.endswith("/"):
                        dir_prefix2 = directory_prefix + "/"
                    file_path_without_prefix = file_path[len(dir_prefix2):]
                    url = "https://" + file_path_without_prefix
                else:
                    url = "file://" + file_path
                
                #content_type = "text/html" # TODO: extract form wget output "Length: 2588 (2,5K) [text/html]"
                content_type = None
                file_size = os.path.getsize(file_path) # OR: # TODO: extract form wget output "Length: 2588 (2,5K) [text/html]"
                logger.info(f"WGET downloaded url: {url} -> file_path: {file_path} (content_type: {content_type}, file_length: {file_size})")
                
                # Construct result Blob
                blob = create_blob_from_local_file(url = url,
                                                   file_path = file_path,
                                                   mimetype = content_type,
                                                   file_size = file_size)
                yield blob




    # Probably not needed anymore:
    """
    @staticmethod
    def _load_text_file(file_path: str) -> Optional[str]:
        default_encoding: Optional[str] = "utf-8"

        text: Optional[str] = None
        try:
            logger.info(f"_load_text_file: {file_path}")
            with open(file_path, "r", encoding=default_encoding) as f:
                text = f.read()
        except UnicodeDecodeError as e:
            logger.info(f"_load_text_file E1: {file_path}")

            detected_encodings = detect_file_encodings(file_path)
            for default_encoding in detected_encodings:
                logger.debug(f"_load_text_file: Trying encoding: {default_encoding.encoding}")
                try:
                    with open(file_path, encoding=default_encoding.encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}") from e
            logger.warning(f"_load_text_file: Error B loading {file_path}: {e}")
            logger.info(f"_load_text_file: ERROR: Error B loading {file_path}: {e}")
            return None
        
        return text
    """
