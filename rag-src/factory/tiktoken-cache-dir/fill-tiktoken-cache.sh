#!/bin/bash

#
# fills the cache (download from https://openaipublic.blob.core.windows.net)
# to be able to run air-gapped afterwards
#
# Details: See https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer
#

export TIKTOKEN_CACHE_DIR=.
python3 -c "import tiktoken; tiktoken.get_encoding('gpt2')"
python3 -c "import tiktoken; tiktoken.get_encoding('r50k_base')"
python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
python3 -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

ls -l

