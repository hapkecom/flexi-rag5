
from functools import cache
from typing import Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from factory.factory_util import call_function_or_constructor
from common.service.configloader import deep_get, settings
import logging

logger = logging.getLogger(__name__)


#
# Specific LLM instances and their setup
#

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


def get_document_grader_chat_llm() -> BaseChatModel:
    config_llm_key = deep_get(settings, "config.rag_response.document_grader_chat_llm")
    llm = setup_llm_for_config_llm_key(config_llm_key)
    logger.info(f"Setup done: config.rag_response.document_grader_chat_llm={llm}")
    return llm


def get_rewrite_question_chat_llm() -> BaseChatModel:
    config_llm_key = deep_get(settings, "config.rag_response.rewrite_question_chat_llm")
    llm = setup_llm_for_config_llm_key(config_llm_key)
    logger.info(f"Setup done: config.rag_response.rewrite_question_chat_llm={llm}")
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

    # Action: Create instance
    return call_function_or_constructor(module_and_class, class_kwargs, context_str_for_logging)

@cache
def get_default_embeddingsOLD() -> Embeddings:
    # TODO: make it configurable XXXXXXXXXXXXXXXXXXXXXXXXXXXx
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()


