
from functools import cache
from typing import Dict, List, Optional

from factory.factory_util import call_function_or_constructor
from common.service.configloader import deep_get, settings
from langchain_core.document_loaders import BaseLoader

from index_builder_and_retrieval_search_service.loader_and_parser.blob_parser_document_loader import BlobParserDocumentLoader
from index_builder_and_retrieval_search_service.loader_and_parser.default_blob_parsers import DefaultBlob2DocumentsParser
from document_loader_service.tools.wget_blob_loader import WgetBlobLoader
from common.utils.string_util import str_limit

import logging

logger = logging.getLogger(__name__)


#
# DocumentLoader instances and their setup
#


@cache
def get_document_loaders() -> List[BaseLoader]:
    """
    Setup all document loaders as specified in the configuration.   
    """

    logger.info("Setup Blob and Document Loaders")
    config_loaders: Dict = deep_get(settings, "config.rag_loading.loaders")

    # Iterate over all keys
    loaders: List[BaseLoader] = []
    loaded_keys = []
    skipped_keys = []
    for key in config_loaders:
        config_loader: Dict = deep_get(config_loaders, key)
        loader = get_configured_loader(config_loader)
        if loader is not None:
            loaders.append(loader)
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    logger.info(f"Setup done: {len(loaders)}/{len(config_loaders)} loaders loaded - loaded: {loaded_keys}, skipped: {skipped_keys}")
    return loaders


def get_configured_loader(config_loader: Dict) -> Optional[BaseLoader]:
    # Start
    context_str_for_logging = str_limit(f"Setup loader: {config_loader}", 400)
    logger.info(context_str_for_logging)

    # Load config
    enabled                     = deep_get(config_loader, "enabled")
    typename                    = deep_get(config_loader, "type")
    module_and_class            = deep_get(config_loader, "class")
    class_kwargs                = deep_get(config_loader, "args")

    # Check if enabled
    if not enabled:
        logger.info(f"Loader is disabled: {config_loader}")
        return None

    # Action: Create instance (depending on type)
    if typename == "PlobLoader":
        plob_loader = call_function_or_constructor(module_and_class, class_kwargs, context_str_for_logging)
        # throw an exception
        raise NotImplementedError("PlobLoader is not implemented yet. Use BlobParserDocumentLoader with BlobLoader instead.")
        document_loader = BlobParserDocumentLoader(blob_loader, DefaultBlob2DocumentsParser()) 
        return document_loader
    if typename == "BlobLoader":
        blob_loader = call_function_or_constructor(module_and_class, class_kwargs, context_str_for_logging)
        document_loader = BlobParserDocumentLoader(blob_loader, DefaultBlob2DocumentsParser()) 
        return document_loader
    elif typename == "BaseLoader":
        document_loader = call_function_or_constructor(module_and_class, class_kwargs, context_str_for_logging)
    else:
        raise ValueError(f"Unknown loader type: {typename} for loader_config: {config_loader}")


#
# TODO: TO DELETE - it was only used during development
#
@cache
def get_wget_document_loader(url: str) -> BaseLoader:

    #
    # Attention: Configuration from yaml is NOT YET USED HERE!!!
    # TODO!!!
    #

    document_loader = BlobParserDocumentLoader(WgetBlobLoader(url), DefaultBlob2DocumentsParser()) 
    return document_loader

    """
    # Start
    config_vectorstore = deep_get(settings, "config.common.databases.vectorstore")
    context_str_for_logging = f"Setup VectorStore: {config_vectorstore}"
    logger.info(context_str_for_logging)

    # Load config
    module_and_class            = deep_get(config_vectorstore, "class")
    class_kwargs                = deep_get(config_vectorstore, "args")
    embedding_function_arg_name = deep_get(config_vectorstore, "embedding_function_arg_name")

    # Add embedding function to class_kwargs
    if embedding_function_arg_name is not None:
        class_kwargs[embedding_function_arg_name] = get_default_embeddings()

    # Action: Create instance
    return call_function_or_constructor(module_and_class, class_kwargs, context_str_for_logging)
    """
