# RAG example
#
# based on
#   https://github.com/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/llamaindex/retrieval-augmented-generation/advanced_rag.ipynb
#

from typing import Callable, List
from importlib.metadata import version
from llama_index.core.response import Response

print(f"LlamaIndex version: {version('llama_index')}")
print(f"Weaviate client version: {version('weaviate-client')}")

import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core.schema import (
    BaseNode,
    #Document,
    #MetadataMode,
    #NodeRelationship,
    #TextNode,
    #TransformComponent,
)

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding()

# loaded data
#   mkdir -p 'input-data'
#   wget 'https://raw.githubusercontent.com/run-llama/llama_index/refs/heads/main/docs/docs/examples/retrievers/data/paul_graham/paul_graham_essay.txt' -O 'input-data/paul_graham_essay.txt'

from llama_index.core import SimpleDirectoryReader

REINDEX_AT_START = False
reindex = REINDEX_AT_START
MAX_TOKEN_FOR_INDEX_EMBEDDINGS = 512  # Set the maximum number of tokens


def parse_nodes(documents) -> List[BaseNode]:
    # Parse documents into nodes
    from llama_index.core.node_parser import SentenceWindowNodeParser

    # Create the sentence window node parser with default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # Extract nodes from documents
    nodes: List[BaseNode] = node_parser.get_nodes_from_documents(documents)

    # Example to showcase what the nodes look like
    i = 10
    print(f"Text: \n{nodes[i].text}")
    print("------------------")
    print(f"Window: \n{nodes[i].metadata['window']}")

    # print JSON of node
    print("------------------")
    print(f"Node JSON: \n{nodes[i].to_dict()}")
    print("------------------")

    return nodes

# Load data

# fill data
file_metadata1: Callable[[str], dict] = lambda x: {"url": f"file:///{x}"}
documents1 = SimpleDirectoryReader(
        input_files=["./input-data/paul_graham_essay.txt"],
        file_metadata=file_metadata1,
).load_data()
# https://en.wikipedia.org/wiki/Java_(programming_language)
file_metadata2: Callable[[str], dict] = lambda x: {"url": "https://en.wikipedia.org/wiki/Java_(programming_language)"}
documents2 = SimpleDirectoryReader(
        input_files=["./input-data/java.html"],
        file_metadata=file_metadata2,
).load_data()
documents = documents1 + documents2

print(f"Number of documents: {len(documents)}")
print(f"Document 0: {documents[0]}")
# print JSON of document
print(f"Document 0 JSON: {documents[0].to_dict()}")

# Parse nodes
nodes = parse_nodes(documents)
print(f"***AAA*** Node metadata: {nodes[0].metadata}")
print(f"***AAA*** Node URL: {nodes[0].metadata.get('url')}")


def open_vectorstore_client_connection():
    # Open logic
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    import weaviate

    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
        headers={
          #  "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Or any other inference API keys
        }
    )
    try:
        print(f"Client: {weaviate.__version__}, Server: {client.get_meta().get('version')}")

        index_name = "MyExternalContext"
        index = None

        # Check if collection exists
        global reindex
        if reindex:
            # Delete the collection if it exists
            print(f"Reindexing - deleting collection {index_name}")
            client.collections.delete(index_name)
            reindex = False

        if client.collections.exists(index_name):
            print(f"Collection {index_name} exists - keep and use it")
            vector_store = WeaviateVectorStore(
                weaviate_client=client,
                index_name=index_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        else:
            print(f"Collection {index_name} does not exist - creating it")
            vector_store = WeaviateVectorStore(
                weaviate_client=client,
                index_name=index_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Limit the input tokens for Weaviate embedding
            truncated_nodes = []
            max_chars_for_index_embeddings = 2*MAX_TOKEN_FOR_INDEX_EMBEDDINGS
            for node in nodes:
                # Truncate the text to fit within the token limit
                if len(node.text) > MAX_TOKEN_FOR_INDEX_EMBEDDINGS:
                    # Truncate the text to fit within the token limit
                    node.text = node.text[:max_chars_for_index_embeddings]
                truncated_nodes.append(node)
            # print JSON of node

            # create index/calculate embeddings
            index = VectorStoreIndex(
              truncated_nodes,
              storage_context=storage_context,
            )

        return client, index
    except Exception as e:
        client.close()
        raise e

import time  # Add import for timing

def init_vectorstore() -> List[BaseNode]:
    print("Creating or opening Weaviate index...")
    start_time = time.time()  # Start timing
    client, index = open_vectorstore_client_connection()
    try:
        print("Created or opened Weaviate index...")
    except Exception as e:
        print(f"Error: {e}")
        return None
    # finally block to ensure the client is closed
    finally:
        client.close()
        duration = time.time() - start_time  # Calculate duration
        print(f"Index creation took {duration:.2f} seconds.")
init_vectorstore()


def search(query: str) -> Response:
    """
    Search for a query in the Weaviate index.
    Args:
        query (str): The search query.
    Returns:
        response (llama_index.core.response.Response): The search response.
    """
    print(f"Search query: {query}")

    # Search logic
    client, index = open_vectorstore_client_connection()
    try:
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor

        postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

        query_engine = index.as_query_engine(
            similarity_top_k=6,
            vector_store_query_mode="hybrid",
            alpha=0.5,
            node_postprocessors=[postproc],
        )

        response = query_engine.query(query)
        print("\n\n------------------")
        print(f"Response: {response}")

        window = response.source_nodes[0].node.metadata["window"]
        sentence = response.source_nodes[0].node.metadata["original_text"]
        print("------------------")
        print(f"Window: {window}")
        print("------------------")
        print(f"Original Sentence: {sentence}")

        return response
    except Exception as e:
        print(f"Error: {e}")
        return None
    # finally block to ensure the client is closed
    finally:
        client.close()


if __name__ == "__main__":
    # Example usage
    search("What happened at Interleaf?")