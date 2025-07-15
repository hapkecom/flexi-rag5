import os
from common.utils.hash_util import sha256sum_file

from typing import Any, Dict, Iterable, Iterator, Optional
from langchain_core.documents.base import Blob as Blob
from langchain_community.document_loaders.helpers import detect_file_encodings
import logging
from datetime import datetime, timezone
import time

logger = logging.getLogger(__name__)


# Construct result Blob
def create_blob_from_local_file(url: str, 
                                file_path: str,
                                mimetype: Optional[str],
                                file_size: Optional[int]) -> Blob:
    #
    # TODO: Improve the retry logic, maybe find the root cause of the issue:
    #
    # """
    # rag-app       | 2025-07-14 20:57:43.992 INFO     wget_blob_loader.py:crawl_with_single_command() - WGET downloaded url: https://hapke.com/index.html -> file_path: /tmp/wget/hapke.com/index.html (content_type: text/html, file_length: 53415)
    # rag-app       | 2025-07-14 20:57:43.994 WARNING  wget_blob_loader.py:crawl_with_single_command() - Command WGET caused en error: [Errno 2] No such file or directory: '/tmp/wget/hapke.com/index.html'
    # rag-app       | 2025-07-14 20:57:43.994 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): Command: 'wget --recursive --compression=auto --header="Authorization: Bearer MY-SECRET-TOKEN" --wait=1 --random-wait --no-parent --no-check-certificate --html-extension --convert-links --level=2 -e robots=off --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-us,en;q=0.5" --reject-regex "favicon\.|css\/|img\/|js\/" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.72 Safari/537.36" --directory-prefix "/tmp/wget" "https://hapke.com/"'
    # rag-app       | 2025-07-14 20:57:43.994 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): --2025-07-14 20:57:43--  https://hapke.com/
    # rag-app       | 2025-07-14 20:57:43.994 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): Resolving hapke.com (hapke.com)... 185.199.110.153, 185.199.111.153, 185.199.109.153, ...
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): Connecting to hapke.com (hapke.com)|185.199.110.153|:443... connected.
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): HTTP request sent, awaiting response... 200 OK
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): Length: 8598 (8.4K) [text/html]
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): Saving to: ‘/tmp/wget/hapke.com/index.html’
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): 
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): 0K ........                                              100% 5.27M=0.002s
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): 
    # rag-app       | 2025-07-14 20:57:43.995 INFO     wget_blob_loader.py:crawl_with_single_command() -     COMMAND(WGET): 2025-07-14 20:57:43 (5.27 MB/s) - ‘/tmp/wget/hapke.com/index.html’ saved [53415]
    # rag-app       | 2025-07-14 20:57:43.996 ERROR    wget_blob_loader.py:crawl_with_single_command() - Exception1:
    # rag-app       | 2025-07-14 20:57:43.996 ERROR    wget_blob_loader.py:crawl_with_single_command() - Traceback (most recent call last):
    # rag-app       |   File "/app/rag-src/document_loader_service/tools/wget_blob_loader.py", line 162, in crawl_with_single_command
    # rag-app       |     blob = create_blob_from_local_file(url = url,
    # rag-app       |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # rag-app       |   File "/app/rag-src/common/blob_creator.py", line 27, in create_blob_from_local_file
    # rag-app       |     "file_last_modified": datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    # rag-app       |                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # rag-app       |   File "<frozen genericpath>", line 55, in getmtime
    # rag-app       | FileNotFoundError: [Errno 2] No such file or directory: '/tmp/wget/hapke.com/index.html'
    # rag-app       | 
    # rag-app       | 2025-07-14 20:57:43.997 ERROR    wget_blob_loader.py:crawl_with_single_command() - Exception2: [Errno 2] No such file or directory: '/tmp/wget/hapke.com/index.html'
    # rag-app       | Traceback (most recent call last):
    # rag-app       |   File "/app/rag-src/document_loader_service/tools/wget_blob_loader.py", line 162, in crawl_with_single_command
    # rag-app       |     blob = create_blob_from_local_file(url = url,
    # rag-app       |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # rag-app       |   File "/app/rag-src/common/blob_creator.py", line 27, in create_blob_from_local_file
    # rag-app       |     "file_last_modified": datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    # rag-app       |                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # rag-app       |   File "<frozen genericpath>", line 55, in getmtime
    # rag-app       | FileNotFoundError: [Errno 2] No such file or directory: '/tmp/wget/hapke.com/index.html'
    # rag-app       | 2025-07-14 20:57:43.998 INFO     wget_blob_loader.py:yield_blobs() - Finished downloading 0 blobs from url='https://hapke.com/', command='wget --recursive --compression=auto --header="Authorization: Bearer MY-SECRET-TO...[627]'
    # """

    # Check if the file exists, retry few times if not found
    max_retries = 10
    sleep_time_seconds = 1
    file_found = False
    for attempt in range(max_retries):
        try:
            x = os.path.getmtime(file_path)
            file_found = True
        except FileNotFoundError:
            logger.info(f"File not found (yet): {file_path} (attempt {attempt + 1}/{max_retries})")
            time.sleep(sleep_time_seconds)
    else:
        if not file_found:
            logger.warning(f"File still not found after {max_retries} attempts: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

    # Create blob from the local file, retry few times if not found
    for attempt in range(max_retries):
        try:
            blob = _create_blob_from_local_file(url, file_path, mimetype, file_size)
            return blob
        except Exception as e:
            logger.info(f"Error creating blob from file '{file_path}' (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(sleep_time_seconds)
    else:
        if not file_found:
            logger.warning(f"Blop still not created after {max_retries} attempts: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")



def _create_blob_from_local_file(url: str, 
                                file_path: str,
                                mimetype: Optional[str],
                                file_size: Optional[int]) -> Blob:

    # Create metadata of the blob
    if file_size is None:
        file_size = os.path.getsize(file_path)
    metadata: Dict[str, Any] = {
        "source": url,
        "file_size": file_size,
        "file_path": file_path,
        "file_sha256": sha256sum_file(file_path),
        "file_last_modified": datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }

    # Create blob
    blob = Blob.from_path(
        path = file_path,
        encoding = _guess_file_encoding(file_path),
        mime_type = mimetype,
        guess_type = True,
        metadata = metadata,
    )

    # Result
    return blob


def _guess_file_encoding(file_path: str) -> Optional[str]:
    try:
        detected_encodings = detect_file_encodings(file_path, timeout=30)
        if len(detected_encodings) > 0:
            # the most likely encoding is the first one
            return detected_encodings[0].encoding
        logger.info(f"Couldn't detecting encoding for '{file_path}'")

    except Exception as e:
        # Handle the exception if needed
        logger.warning(f"Error detecting encoding for '{file_path}': {e}")
        # You can log the error or raise an exception if necessary
    
    # Fallback to utf-8
    default_encoding = "utf-8"
    return default_encoding
