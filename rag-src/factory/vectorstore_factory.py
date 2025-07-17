
from functools import cache
from langchain_core.vectorstores import VectorStore
import json
from typing import (
    Any,
    List,
    Optional,
)

import weaviate
# doesn't support get_by_ids(): 
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.client import WeaviateClient
#from langchain_community.vectorstores.weaviate import Weaviate

from factory.factory_util import call_function_or_constructor
from factory.llm_factory import get_default_embeddings
from index_builder_basics.embeddings_cache import get_cached_default_embeddings
from common.service.configloader import deep_get, settings
import logging

logger = logging.getLogger(__name__)


#
# vectorstore instance and its setup
#

collection_prefix = "weaviate_vectorstore_"

# TODO: CONFIG CURRENTLY NOT USED - Vectorstore is hard-coded in factory.vectorstore_factory.py!!!
@cache
def get_vectorstore_NOT_USED() -> VectorStore:
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

def get_vectorstore_stats(log_all_entries: bool = False) -> str:
    weaviate_client = get_weaviate_client()
    global collection_prefix

    if weaviate_client:
        if weaviate_client.collections:
            collection_name = get_vectorstore_collection_name()
            # Check if the collection exists
            collection_exists = weaviate_client.collections.exists(collection_name)
            if not collection_exists:
                return f"Collection '{collection_name}' does not exist."
            # Action
            objs = weaviate_client.collections.get(collection_name)
            result = ""
            if log_all_entries:
                result +=     f"Collection '{collection_name}' contains the following {len(objs)} objects:\n"
                for obj in objs.iterator(include_vector=False):
                    result += f"    Object ID: {obj.uuid}, Properties: {obj.properties}\n"
            else:
                result = f"Collection '{collection_name}' contains {len(objs)} objects."
            return result
        else:
            return "No (Weaviate) collection available."
    else:
        return "No (Weaviate) client available."

def print_vectorstore_stats() -> None:
    if logger.isEnabledFor(logging.DEBUG):
        stats = get_vectorstore_stats(True)
        logger.debug(stats)
    else:
        stats = get_vectorstore_stats(False)
        logger.info(stats)


get_weaviate_client_was_called = False

@cache
def get_weaviate_client(log_metadata = False) -> WeaviateClient:
    """Get the Weaviate client."""

    #global weaviate_client
    global collection_prefix
    global get_weaviate_client_was_called

    # Load config (preparation)
    config_vectorstore_database = deep_get(settings, "config.common.databases.vectorstore")
    context_str_for_logging = f"Setup vectorstore connection: {config_vectorstore_database}"
    logger.info(context_str_for_logging)
    # Load config
    weaviate_host = deep_get(config_vectorstore_database, "weaviate_host")
    weaviate_port = deep_get(config_vectorstore_database, "weaviate_port")
    weaviate_grpc_port = deep_get(config_vectorstore_database, "weaviate_grpc_port")

    weaviate_client = weaviate.connect_to_local(
        #host="127.0.0.1",
        #port=8080,
        #grpc_port=50051,
        host=weaviate_host,
        port=weaviate_port,
        grpc_port=weaviate_grpc_port,
    )
    logger.info(f"weaviate_client.is_ready()={weaviate_client.is_ready()}")

    # Get meta information using the Weaviate client
    if log_metadata or not get_weaviate_client_was_called:
        meta_info = weaviate_client.get_meta()
        meta_info['modules'] = "..."
        all_collections = weaviate_client.collections.list_all(simple = True)
        # Print meta information
        logger.info(f"Weaviate Meta Information: {json.dumps(meta_info, indent=2)}")
        logger.info(f"Weaviate all_collections: {all_collections}")

    get_weaviate_client_was_called = True
    return weaviate_client

# def get_newest_collection_name() -> Optional[str]:
#     """Get the newest existing collection name."""

#     weaviate_client = get_weaviate_client()
#     all_collections = weaviate_client.collections.list_all(simple = True)
#     all_relevant_collections: List[str] = []

#     # iterate over keys in a Dict d
#     for collection_name in all_collections:
#         logger.debug(f"collection_name='{collection_name}'")
#         if collection_name.startswith(collection_prefix):
#             all_relevant_collections.append(collection_name)

#     # get the newest collection = highest name
#     if len(all_relevant_collections) > 0:
#         all_relevant_collections.sort(reverse=True)
#         newest_collection_name = all_relevant_collections[0]
#         return newest_collection_name
#     else:
#         logger.warning("No collection found")
#         return None

# def createget_newest_collection_name() -> Optional[str]:
#     # TODO XXXXXXXXXXXXXXXXXXXXXX: Transactionssicherheit beim collection-Wechsel!!!

def get_vectorstore_collection_name() -> str:
    # Build name
    embedding_model_id = deep_get(settings, "config.common.embedding_model_id")
    collection_suffix = embedding_model_id
    collection_name = f"{collection_prefix}-model-{collection_suffix}"

    # replace all non-alphanumeric characters with an underscore
    collection_name = ''.join(c if c.isalnum() else '_' for c in collection_name)
    # remove leading and trailing underscores
    collection_name = collection_name.strip('_')

    return collection_name

@cache
def get_vectorstore() -> VectorStore:
    # TODO: Remove hard-coded vectorstore

    #from langchain_chroma import Chroma
    #vector_store = Chroma(
    #    collection_name="example_collection",
    #    embedding_function=embeddings,
    #    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary


    # Create a VectorStore instance
    collection_name = get_vectorstore_collection_name()

    vector_store = WeaviateVectorStore(
        client=get_weaviate_client(),
        index_name=collection_name,
        text_key="text",
        embedding= get_cached_default_embeddings()
    )

    # add function "get_by_ids" to vectors_store
    # def get_by_ids(self, ids):
    #     """
    #     Get documents by their IDs.
    #     """
    #     my_collection = weaviate_client.collections.get(collection_name)
    #     objs = []
    #     for id in ids:
    #         logger.debug(f"get_by_ids: id='{id}'")
    #         obj = my_collection.query.fetch_object_by_id(id)
    #         if obj is None:
    #             # raise ValueError(f"Object with ID {id} not found in collection {collection_name}.")
    #             logger.warning(f"Object with ID {id} not found in collection {collection_name}.")
    #         if obj is not None:
    #             objs.append(obj)
    #     return [ obj ]
    # vector_store.get_by_ids = get_by_ids.__get__(vector_store, WeaviateVectorStore)
    return vector_store


def clean_vectorstore(index_build_id: str) -> int:
    """
    Iterate through all entries in the vectorstore and delete entries
    that don't have the specified metadata key/value pair.

    Args:
        index_build_id (str): The index build ID to check against.
    Returns:
        int: The number of deleted objects.
    """
    weaviate_client = get_weaviate_client()
    collection_name = get_vectorstore_collection_name()
    metadata_key = "index_build_id"
    metadata_value_expected = index_build_id

    # Check if the collection exists
    collection_exists = weaviate_client.collections.exists(collection_name)
    if not collection_exists:
        logger.warning(f"Collection '{collection_name}' does not exist.")
        return 0
    logger.info(f"Collection '{collection_name}' found.")

    # Fetch all objects in the collection
    collection = weaviate_client.collections.get(collection_name)

    counter_all = 0
    counter_to_delete = 0
    counter_deleted = 0
    for item in collection.iterator(include_vector=False):
        counter_all += 1
        # Check if the metadata key/value pair exists
        metadata = item.properties
        #logger.debug(f"Object Metadata: {metadata}")
        metadata_value_actual = metadata.get(metadata_key)
        if metadata_value_actual != metadata_value_expected:
            # Delete the object if it doesn't match the criteria
            counter_to_delete += 1
            item_uuid = item.uuid
            if item_uuid:
                logger.debug(f"Deleting item with UUID '{item_uuid}' with {metadata_key}='{metadata_value_actual}' from collection '{collection_name}'.")
                #weaviate_client.collections.delete_object(collection_name, obj_id)
                try:
                    # Delete the object using the Weaviate client
                    collection.data.delete_by_id(item_uuid)
                    counter_deleted += 1
                except Exception as e:
                    logger.warning(f"Error deleting object with ID '{item_uuid}' from collection '{collection_name}': {e}")
            else:
                logger.warning(f"Object without ID found in collection '{collection_name}' with metadata={metadata}, skipping deletion.")
        else:
            logger.debug(f"Object with ID '{item.uuid}' has {metadata_key}='{metadata_value_actual}', keeping it in collection '{collection_name}'.")

    logger.info(f"Deleted {counter_deleted}/{counter_to_delete} of {counter_all} items from collection '{collection_name}' with {metadata_key}='{metadata_value_expected}'.")
    return counter_deleted
