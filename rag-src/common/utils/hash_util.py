import hashlib
import logging
from .string_util import str_limit

logger = logging.getLogger(__name__)

# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

#md5 = hashlib.md5()
#sha1 = hashlib.sha1()


def sha256sum_file(file_path: str) -> str:
    """Compute the SHA-256 hash of a
       file at the given path."""
    
    # start with a fresh buffer
    sha256 = hashlib.sha256()

    # read and process file content
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            #md5.update(data)
            #sha1.update(data)
            sha256.update(data)
    #hexdigest = md5.hexdigest()
    #hexdigest = sha1.hexdigest()
    hexdigest = sha256.hexdigest()

    #print(f"md5sum({file_path}): {hexdigest}")
    #print(f"sha1sum({file_path}): {hexdigest}")
    logger.debug(f"sha256sum_file({file_path}): {hexdigest}")
    return hexdigest


def sha256sum_str(s: str) -> str:
    """Compute the SHA-256 hash of a string."""

    # start with a fresh buffer
    sha256 = hashlib.sha256()

    # process string
    sha256.update(s.encode('utf-8'))
    result = sha256.hexdigest()
    
    logger.debug(f"sha256sum_str('{str_limit(s)}' (len={len(s)})) = {result}")
    return result


# for manual testing:
#filename = "/tmp/wget/dance123.org/index.html"
#sha256sum(filename)
