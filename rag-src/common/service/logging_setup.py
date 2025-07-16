import logging

def setup_logging():
    """
    Setup logging for all modules of this project.
    """

    default_loglevel = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(filename)s:%(funcName)s() - %(message)s',
        level=default_loglevel,
        datefmt='%Y-%m-%d %H:%M:%S')

    #
    # Set different levels for different modules
    #
    logging.getLogger('api').setLevel(logging.INFO)
    logging.getLogger('common').setLevel(logging.INFO)
    logging.getLogger('common.utils').setLevel(logging.INFO)
    logging.getLogger('document_loader_service').setLevel(logging.DEBUG)
    logging.getLogger('document_loader_service.tools').setLevel(logging.INFO)
    #logging.getLogger('document_loader_service.tools.wget_blob_loader').setLevel(logging.DEBUG)
    logging.getLogger('index_builder_and_retrieval_search_service').setLevel(logging.DEBUG)
    logging.getLogger('index_builder_and_retrieval_search_service.build_index').setLevel(logging.INFO)
    logging.getLogger('index_builder_and_retrieval_search_service.loader_and_parser').setLevel(logging.INFO)
    logging.getLogger('index_builder_and_retrieval_search_service.document_retrieval').setLevel(logging.INFO)
    logging.getLogger('index_builder_and_retrieval_search_service.document_retrieval_grader').setLevel(logging.INFO)
    logging.getLogger('index_builder_and_retrieval_search_service.document_storage').setLevel(logging.INFO)
    logging.getLogger('index_builder_and_retrieval_search_service.document_summarizer').setLevel(logging.INFO)
    logging.getLogger('index_builder_basics').setLevel(logging.DEBUG)
    logging.getLogger('index_builder_basics.embeddings_cache').setLevel(logging.INFO)
    logging.getLogger('model').setLevel(logging.INFO)
    logging.getLogger('rag_chat_service').setLevel(logging.INFO)

    logging.getLogger('test').setLevel(logging.DEBUG)
    logging.getLogger('test_data_gen').setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)

    #
    # Suppress specific warnings
    #

    # Example: Filter out the websockets deprecation warning until dependencies are updated
    # This addresses: DeprecationWarning: websockets.server.WebSocketServerProtocol is deprecated
    #warnings.filterwarnings("ignore", message=".*websockets.server.WebSocketServerProtocol.*", category=DeprecationWarning)


    #
    # Done
    #
    logger.info("Logging setup done")


"""
Alternative (not tested yet) logging configuration using dictConfig:


import logging
import logging.config

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s %(name)-30s %(levelname)-8s %(message)s"
        },
        "simple": {
            "format": "%(levelname)-8s %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file_debug", "file_error"],
            "level": "DEBUG",
            "propagate": False
        },
        "myapp.module_b": {
            "level": "WARNING",
            # erbt Handler des root
        }
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING)

"""