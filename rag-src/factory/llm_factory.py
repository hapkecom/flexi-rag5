
from functools import cache
from typing import Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
import os

from factory.factory_util import call_function_or_constructor
from common.service.configloader import deep_get, settings
import logging
from common.utils.string_util import str_limit

logger = logging.getLogger(__name__)

from langchain_ollama import ChatOllama

#
# Specific LLM instances and their setup
#

@cache
def get_document_summarizer_chat_llm() -> BaseChatModel:
    config_llm_key = deep_get(settings, "config.rag_indexing.document_summarizer_chat_llm")
    llm = setup_llm_for_config_llm_key(config_llm_key)
    logger.info(f"Setup done: config.rag_indexing.document_summarizer_chat_llm={llm}")
    return llm



@cache
def get_document_grader_chat_llm() -> BaseChatModel:
    config_llm_key = deep_get(settings, "config.rag_response.document_grader_chat_llm")
    llm = setup_llm_for_config_llm_key(config_llm_key)
    logger.info(f"Setup done: config.rag_response.document_grader_chat_llm={llm}")
    return llm

@cache
def get_rewrite_question_chat_llm() -> BaseChatModel:
    config_llm_key = deep_get(settings, "config.rag_response.rewrite_question_chat_llm")
    llm = setup_llm_for_config_llm_key(config_llm_key)
    logger.info(f"Setup done: config.rag_response.rewrite_question_chat_llm={llm}")
    return llm



@cache
def get_default_chat_llm_without_streaming() -> BaseChatModel:
    config_llm_key = deep_get(settings, "config.rag_response.default_chat_llm")
    llm = setup_llm_for_config_llm_key(config_llm_key)
    logger.info(f"Setup done: config.rag_response.default_chat_llm={llm}")
    return llm

@cache
def get_default_chat_llm_with_streaming() -> BaseChatModel:
    config_llm_key = deep_get(settings, "config.rag_response.default_chat_llm_with_streaming")
    llm = setup_llm_for_config_llm_key(config_llm_key)
    logger.info(f"Setup done: config.rag_response.default_chat_llm_with_streaming={llm}")
    return llm

#
# Helper functions for dynamic LLM setup
#

def setup_llm_for_config_llm_key(config_llm_key: str) -> Optional[BaseChatModel]:
    logger.info(f"Setup LLM from config_llm_key: {config_llm_key}")
    llm_config = deep_get(settings, f"config.common.chat_llms.{config_llm_key}")
    return setup_llm_for_config(config_llm_key=config_llm_key, llm_config=llm_config)

def setup_llm_for_config(
        config_llm_key: Optional[str],
        llm_config: Optional[Dict],
    ) -> BaseChatModel:
    """
    Setup LLM from config. Fails with error if not possible.

    Args:
        config_llm_key (str): LLM name (for logging purposes only)
        llm_config (dict): LLM configuration
        
    Returns:
        BaseChatModel: LLM instance, None in the case of an error
    """

    # TODO: remove access tokens before logging???
    context_str_for_logging = f"LLM setup: {config_llm_key}({llm_config})"
    logger.info(context_str_for_logging)

    # Pre-checks
    if llm_config is None:
        logger.error(f"Error in LLM setup: llm_config is None for {config_llm_key}")
        return None

    # Load config
    module_and_class = deep_get(llm_config, "class")
    class_kwargs = deep_get(llm_config, "args")

    # add args to class_kwargs - TODO: make it more generic, currently is's only a hack for Ollama
    auth = deep_get(class_kwargs, "kwargs_header_authorization", None)
    if auth is not None:
        class_kwargs["client_kwargs"] = {
            "headers": {
                "Authorization": auth
            }
        }
        # remove the auth from class_kwargs
        del class_kwargs["kwargs_header_authorization"]

    # Action: Create instance
    return call_function_or_constructor(module_and_class, class_kwargs, context_str_for_logging)


#
# Specific embedding-model instances and their setup
#

@cache
def get_default_embeddings() -> Embeddings:
    # Start
    config_embedding_llm = deep_get(settings, "config.common.embedding_llm")
    context_str_for_logging = f"Setup Embedding LLM: {config_embedding_llm}"
    logger.info(context_str_for_logging)

    # Load config
    module_and_class = deep_get(config_embedding_llm, "class")
    class_kwargs = deep_get(config_embedding_llm, "args")

    # add args to class_kwargs - TODO: make it more generic, currently is's only a hack for Ollama
    auth = deep_get(class_kwargs, "kwargs_header_authorization", None)
    if auth is not None:
        class_kwargs["client_kwargs"] = {
            "headers": {
                "Authorization": auth
            }
        }
        # remove the auth from class_kwargs
        del class_kwargs["kwargs_header_authorization"]

    # Action: Create instance
    return call_function_or_constructor(module_and_class, class_kwargs, context_str_for_logging)

@cache
def get_default_embeddingsOLD() -> Embeddings:
    # TODO: make it configurable XXXXXXXXXXXXXXXXXXXXXXXXXXXx
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()


#
# Helper function to text LLM connection
#
def test_llm_connection(llm: BaseChatModel, info: str) -> bool:
    """
    Test the connection to the LLM by sending a simple prompt.
    
    Args:
        llm (BaseChatModel): The LLM instance to test.
        info (str): Additional information for logging.
        
    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    try:
        logger.debug(          f"LLM connection test of '{info}' starts...")
        response = llm.invoke("Hello, how are you?")
        logger.debug(str_limit(f"LLM connection test of '{info}' successful: {response}", 300))
        return True
    except Exception as e:
        logger.error(f"LLM connection test of '{info}' failed: {e}", exc_info=True)
        return False

def test_embeddings_connection(embeddings: Embeddings, info: str) -> bool:
    """
        Test the connection to the embeddings model by generating embeddings for a simple text.
        
        Args:
            embeddings (Embeddings): The embeddings instance to test.
            info (str): Additional information for logging.

        Returns:
            bool: True if the connection is successful, False otherwise.
    """
    try:
        logger.debug(f"Embeddings connection test of '{info}' starts...")
        response = embeddings.embed_query("Hello, how are you?")
        logger.debug(str_limit(f"Embeddings connection test of '{info}' successful: {response}", 300))
        return True
    except Exception as e:
        logger.error(f"Embeddings connection test of '{info}' failed: {e}", exc_info=True)
        return False

def test_all_llm_and_embedding_llm_connections() -> bool:
    """
    Test all LLM connections defined in the configuration.
    
    Returns:
        bool: True if all connections are successful, False otherwise.
    """
    all_successful = True

    # Preparation
    if not init_tiktiken_cache():
        all_successful = False

    # Test each LLM connection
    all_chat_llms = [
        # ignored here because not very relevant: (get_default_chat_llm_with_streaming(), "default_chat_llm_with_streaming"),
        # ignored here because not very relevant: (get_default_chat_llm_without_streaming(), "default_chat_llm_without_streaming"),
        (get_document_grader_chat_llm(), "document_grader_chat_llm"),
        (get_rewrite_question_chat_llm(), "rewrite_question_chat_llm"),
    ]
    for llm, info in all_chat_llms:
        if not test_llm_connection(llm, info):
            all_successful = False

    # Test each embedding LLM connection
    all_embeddings = [
        (get_default_embeddings(), "default_embeddings")
    ]
    for embeddings, info in all_embeddings:
        if not test_embeddings_connection(embeddings, info):
            all_successful = False

    return all_successful


#
#
#

@cache
def init_tiktiken_cache() -> bool:
    """
    Initialize the tiktoken cache
    to avoid downloads from https://openaipublic.blob.core.windows.net.

    Background: see
    - /tiktoken-cache-dir/fill-tiktoken-cache.sh
    - https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer

    
    Returns:
        bool: True if the cache is initialized successfully, False otherwise.
    """
    try:
        # set tokenizer cache temporarily
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            should_revert = True
            os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tiktoken-cache-dir",
            )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize tiktoken cache: {e}")
        return False
 