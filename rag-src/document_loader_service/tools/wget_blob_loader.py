import os
from typing import Any, Dict, Iterable, Iterator, Optional
#from langchain_community.document_loaders.blob_loaders.file_system import FileSystemBlobLoader
#from langchain_community.document_loaders.[blob_loaders.schema] import BlobLoader
from langchain_community.document_loaders import BlobLoader
from langchain_core.documents.base import Blob as Blob
from common.utils.string_util import str_limit

import logging

from common.blob_creator import create_blob_from_local_file

logger = logging.getLogger(__name__)


class WgetBlobLoader(BlobLoader):
    def __init__(self,
                 url: Optional[str] = None,
                 max_files: int = -1,
                 command: Optional[str] = None,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the WgetBlobLoader with a URL or a command."""

        self.url = url
        self.max_files = max_files
        self.command = command

        if not self.url and not self.command:
            raise ValueError("Either 'url' or 'command' must be provided to WgetBlobLoader.")

    def yield_blobs(
        self,
    ) -> Iterable[Blob]:
        """Yield blobs that match the requested pattern."""
        
        # Crawl with wget and iterate over the blobs (downloaded files)
        logger.info(f"Downloading files from url='{self.url}', command='{str_limit(self.command, 80)}")
        blobs: Iterator[Blob] = []
        if self.command is None and self.url is None:
            raise ValueError("Either 'url' or 'command' must be provided to WgetBlobLoader.")
        if self.command is None:
            # Use the url to crawl with wget
            logger.debug(f"Using URL '{self.url}' for crawling with wget")
            blobs = WgetBlobLoader.crawl_single_url_with_wget(self.url)
        else:
            # Use the command to crawl with wget
            logger.debug(f"Using command for crawling with wget")
            command = self.command
            
            # Tune command
            if self.url is not None:
                # Replace {url} in command with the actual url
                command = command.replace("{url}", self.url)
            directory_prefix = "/tmp/wget"
            if directory_prefix is not None:
                # Replace {directory_prefix} in command with the actual directory_prefix
                command = command.replace("{directory_prefix}", directory_prefix)
            
            # Exec command
            blobs = WgetBlobLoader.crawl_with_single_command(command, directory_prefix, self.max_files)

        count = 0
        for blob in blobs:
            # extract text from downloaded blob
            logger.debug(f"Downloaded file: {blob}")
            count += 1
            yield blob

        logger.info(f"Finished downloading {count} blobs from url='{self.url}', command='{str_limit(self.command, 80)}'")


    def __str__(self) -> str:
        return f"WgetBlobLoader(url: {self.url})"


    @staticmethod
    def crawl_single_url_with_wget(url) -> Iterator[Blob]:
        directory_prefix = "/tmp/wget"
        command = f"wget --directory-prefix {directory_prefix} --recursive -l1 --no-parent -A.html,.txt,.mp4,.pdf --limit-rate=1024k --wait=10 {url}"
        return WgetBlobLoader.crawl_with_single_command(command, directory_prefix)

    @staticmethod
    def crawl_with_single_command(command, directory_prefix, max_files = None) -> Iterator[Blob]:
        # crawl with command(wget) with popen
        # and read name of downloaded files from stdin/stdout (with popen)
        import subprocess
        import os
        import io

        logger.debug(f"Crawl now with (wget compatible) command:    {command}")
        commandStr = "WGET"

        # Collect outpout fromt stdout in array
        output_lines = [f"Command: '{command}'"]
        downloaded = False
        file_count = 0

        # Run the command
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            shell=True
        ) as proc:
            try:
               with io.TextIOWrapper(proc.stderr, encoding="utf-8") as stderr_wrapper:
                    content_type = None
                    for line in stderr_wrapper:  # or another encoding
                        # Do something with line
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"    COMMAND({commandStr}): {line.strip()}")
                        else:
                            output_lines.append(line.strip())

                        # Check the max_blobs limit
                        if max_files is not None and max_files > 0 and file_count >= max_files:
                            logger.info(f"Reached max_files limit: {max_files}. Stopping further processing.")
                            # kill the process
                            proc.kill()
                            break
                        
                        # Extract content type
                        # from wget output "Length: 2588 (2,5K) [text/html]"
                        if "Length:" in line:
                            try:
                                # Extract content type from the line
                                startstr = "["
                                endstr = "]"
                                if startstr in line and endstr in line:
                                    content_type = line.split(startstr)[1].split(endstr)[0]
                                    logger.debug(f"Extracted content_type: {content_type}")
                            except Exception as e:
                                logger.warning(f"Error extracting content_type from line: {line.strip()} - {e}")

                        else:
                            # Extract <file_path> from line with "'<file_path>' saved"
                            # example line: 2025-07-02 15:30:24 (3.95 MB/s) - ‘example.com/index.html’ saved [53415/53415]
                            # cleanup single ticks
                            defaulttick = "'"
                            line = line.replace("‘", defaulttick).replace("’", defaulttick)
                            startstr = f"{defaulttick}"
                            endstr = f"{defaulttick} saved"
                            if startstr in line and endstr in line:
                                # Extract file_path from line
                                file_path = line.split(startstr)[1].split(endstr)[0]
                                # Derive url from file_path
                                if file_path.startswith(directory_prefix):
                                    dir_prefix2 = directory_prefix
                                    if not file_path.endswith("/"):
                                        dir_prefix2 = directory_prefix + "/"
                                    file_path_without_prefix = file_path[len(dir_prefix2):]
                                    url = "https://" + file_path_without_prefix
                                else:
                                    url = "file://" + file_path

                                # Get details from downloaded file
                                file_size = os.path.getsize(file_path) # OR: # TODO: extract form wget output "Length: 2588 (2,5K) [text/html]"
                                logger.info(f"{commandStr} downloaded url: {url} -> file_path: {file_path} (content_type: {content_type}, file_length: {file_size})")

                                # Construct result Blob
                                blob = create_blob_from_local_file(url = url,
                                                                file_path = file_path,
                                                                mimetype = content_type,
                                                                file_size = file_size)
                                downloaded = True
                                file_count += 1
                                yield blob

                    if not downloaded:
                        # no file was downloaded
                        logger.warning(f"Command {commandStr} did not download any files")
                        for line in output_lines:
                            logger.info(f"    COMMAND({commandStr}): {line.strip()}")
            except Exception as e:
                logger.warning(f"Command {commandStr} caused en error: {e}")
                for line in output_lines:
                    logger.info(f"    COMMAND({commandStr}): {line.strip()}")

                # log full stack trace of the exception e
                logger.warning(f"Exception: {e}", exc_info=True)

                # Exit the Python process with an error code
                # ONLY DO THIS FOR TESTING wget_blob_loader.py:
                #import sys
                #sys.exit(1)


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
