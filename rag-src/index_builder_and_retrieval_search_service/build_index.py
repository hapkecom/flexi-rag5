### Build Index

# partitially based on idea of
#   https://stackoverflow.com/questions/52534211/python-type-hinting-with-db-api/77350678#77350678
from typing import TYPE_CHECKING
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)
import shortuuid
import mimetypes
from factory.document_loader_factory import get_document_loaders
from langchain_core.document_loaders import BaseLoader
from factory.vectorstore_factory import get_vectorstore, clean_vectorstore
from datetime import datetime, timezone

import threading
import time
from langchain_core.vectorstores import VectorStore
from common.service.configloader import deep_get, settings
from factory.llm_factory import test_all_llm_and_embedding_llm_connections
from langchain_core.documents import Document
import logging

from common.utils.hash_util import sha256sum_str
from common.utils.string_util import str_limit
import queue

from .document_storage import save_single_plob_and_its_documents_in_databases
from index_builder_basics.document_storage_sql_database import print_all_from_sqldb
from factory.vectorstore_factory import print_vectorstore_stats
from .document_splitter import split_single_document_into_parts_if_needed
from common.plob_creator import create_virtual_plob
from model.plob import Plob

logger = logging.getLogger(__name__)

#sqlCon: DBAPIConnection | None = None
sqlCon = None
vectorStore: Optional[VectorStore] = None
vectorStoreRetriever = None

# in-memory queue of downloaded plocs with documents to process
downloadedPlogsToProcessQueue = queue.Queue()




indexing_single_run_counter = 0

def get_indexing_single_run_counter():
    global indexing_single_run_counter
    return indexing_single_run_counter

# main function,
# wait/block until for the first round to finish
def start_indexing():
    # start new thread with endless loop

    # turn-on the worker thread
    threading.Thread(target=indexing_endless_loop_worker, daemon=False).start()

    # wait for the first round to finish
    #while get_indexing_single_run_counter() < 1:
    #    time.sleep(1)
    # first round done
    #logger.info("First indexing round done. Indexing is now running in the background.")


def indexing_endless_loop_worker():
    load_every_seconds = deep_get(settings, "config.rag_loading.load_every_seconds")

    while True:
        # preparation
        starttime = time.time()

        # action
        if test_all_llm_and_embedding_llm_connections():
            logger.info("===== All LLM and embedding connections are working. Starting indexing run ...")
            indexing_single_run()
        else:
            logger.error("===== One or more LLM or embedding connections are not working. Skipping indexing run ...")

        # finish this round
        now = time.time()
        seconds_until_next_run = int(load_every_seconds - (now - starttime))
        if seconds_until_next_run > 0:
            logger.info(f"Sleeping for {seconds_until_next_run} seconds before starting next indexing round ...")
            time.sleep(seconds_until_next_run)
        else:
            logger.info(f"Indexing round took longer than {load_every_seconds} seconds. Starting next indexing round immediately ...")




def indexing_single_run():
    global indexing_single_run_counter

    # "index_build_id" to identify this run,
    # and to delete data from older runs from vectorStore
    now = datetime.now(timezone.utc).isoformat()
    index_build_id = f"build_index_run_{now}"

    logger.info(f"===== START (#{indexing_single_run_counter}, '{index_build_id}') =====")

    """
    Here we decouple the crawling/loading and the processing/saving of the downloaded documents
    by using a queue and separated threads.
    """

    # start the worker thread to crawl/load all documents
    threading.Thread(target=download_all_documents_and_put_them_into_queue, args=(), daemon=False).start()


    # process all documents from the queue
    process_all_plobs_from_queue_worker(index_build_id)

    # wait until all documents are completely processed - needed before we can clean the vectorStore
    downloadedPlogsToProcessQueue.join()
    logger.info(f"===== ALL DOCUMENTS PROCESSED (#{indexing_single_run_counter}, '{index_build_id}') =====")

    # cleanup of vectorStore
    logger.info(f"===== RESULTS BEFORE CLEANUP (#{indexing_single_run_counter}, '{index_build_id}') =====")
    print_vectorstore_stats()
    clean_vectorstore(index_build_id)
    logger.info(f"===== RESULTS AFTER CLEANUP (#{indexing_single_run_counter}, '{index_build_id}') =====")
    print_vectorstore_stats()

    # single run done
    logger.info(f"===== RESULTS (#{indexing_single_run_counter}, '{index_build_id}') =====")
    print_all_from_sqldb()
    print_vectorstore_stats()
    #vectorStore = get_vectorstore()
    #logger.info(f"vectorStore = {vectorStore}")
    logger.info(f"===== END (#{indexing_single_run_counter}, '{index_build_id}') =====")
    indexing_single_run_counter += 1


def download_all_documents_and_put_them_into_queue():
    logger.info(f"== download_all_documents_and_put_them_into_queue(): Loading ...")
    document_loaders: List[BaseLoader] = get_document_loaders()
    for document_loader in document_loaders:
        # single document loader (from the config) to process
        document_loader_info_str = str(document_loader)
        # Load plobs or documents
        if hasattr(document_loader, "lazy_load_plobs"):
            # Load plobs directly
            logger.info(f"== Lazy loading plobs using ... {document_loader_info_str}")
            plobs = document_loader.lazy_load_plobs()
            put_downloaded_plobs_into_queue(document_loader_info_str, plobs)
            logger.info(f"== Lazy loading plobs + putting into queue plobs ... {str_limit(plobs, 1024)}")
        else:
            # Load documents
            logger.info(f"== Lazy loading plob with documents using ... {document_loader_info_str}")
            docs = document_loader.lazy_load()
            plob = create_single_plob_from_document_loader(document_loader_info_str, list(docs))
            put_downloaded_plobs_into_queue(document_loader_info_str, [plob])
            logger.info(f"== Lazy loading plob with documents + putting into queue plob with documents ... {str_limit(docs, 1024)}")

    # Add end signal to queue to finisj this loading round
    downloadedPlogsToProcessQueue.put(None)

def create_single_plob_from_document_loader(document_loader_info_str: str, documents: List[Document]) -> Plob:
    """
    Create a plob from a document loader and its documents -
    Only if the document loader doesn't support plobs.
    """
    # Plob
    plob = create_virtual_plob(document_loader_info_str)
    plob.documents = documents
    return plob


def put_downloaded_plobs_into_queue(context_str: str, plobs: Iterator[Plob]):
    logger.info(f"== START {context_str} ...")
    counter = 0
    for plob in plobs:
        downloadedPlogsToProcessQueue.put(plob)
        counter += 1
    logger.info(f"== END {context_str} ... after {counter} documents")


def process_all_plobs_from_queue_worker(index_build_id: str):
    logger.info("== Split and save documents in databases - START")

    while True:
        # get next document from queue
        logger.info(f"Next doc from queue: Take it now (queue len={downloadedPlogsToProcessQueue.qsize()}) ... (blocking) ...")
        plob = downloadedPlogsToProcessQueue.get()
        if plob is None:
            # end signal
            logger.info("Next plob from queue: No more plobs/documents in queue (and no more will come)")
            downloadedPlogsToProcessQueue.task_done()
            break
        else:
            # normal processing
            logger.info(f"Next plob with documents from queue: Got it")
            process_single_plob_and_store_results_in_databases(index_build_id, plob)
            downloadedPlogsToProcessQueue.task_done()

    logger.info("== Split and save documents in databases - END")

#
# processing multiple documents
#

#
# Attention: We need to iterate oder configured loaders instead of URLs
# TODO!!!
#
#def run_all_document_loaders() -> Iterator[Document]:
#    for url in urls:
#        document_loader = get_document_loader(url)
#        docs_of_single_url = document_loader.lazy_load()
#        yield from docs_of_single_url



#
# processing a single plob and its documents
#

def process_single_plob_and_store_results_in_databases(index_build_id: str, plob: Plob):
    """
    Process (load and split) a single plob and its documents
    and store (and index) the results in the SQL DB and the vectorstore.

    NOT LAZY: The plob is processed and saved in the SQL DB and the vectorstore.
    """

    logger.info(f"plob={str_limit(plob)} ...")

    # Get documents from plob and split them into parts if needed
    documents = plob.documents
    if not documents:
        logger.info(f"plob={str_limit(plob)} ... no documents to process")
        return
    # One or multiple documents available
    splited_documents = []
    for doc in documents:
        doc_splits = split_single_document_into_parts_if_needed(doc)
        splited_documents.extend(doc_splits)
        doc = documents[0]
        logger.info(f"plob={str_limit(plob)} ... only one document to process")
        # split document into parts

    # Enrich documents with metadata
    doc_splits = _enrich_plob_documents(index_build_id, plob, doc_splits)
    doc_splits = list(doc_splits)  # convert to list to allow multiple iterations
    logger.info(f"Documents of the plob: ")
    for doc in doc_splits:
        logger.info(f"doc.metadata={str_limit(doc.metadata, 1024)} doc.page_content={str_limit(doc.page_content)}")

    # Save plob in SQL DB and in vectorstore
    save_single_plob_and_its_documents_in_databases(plob, doc_splits)

    logger.info(f"doc={str_limit(plob)} ... DONE")



def _enrich_plob_documents(index_build_id: str,
                           plob: Plob,
                           document: Iterator[Document]
                          ) -> Iterator[Document]:
    """
    Enrich a documents (part of a plob) with metadata.

    Args:
        index_build_id: The index build ID to identify this run.
        plob: The (parent) plob to which the documents belongs.
        documents: The documents to be enriched.
    """
    # iterate over all contents
    for doc in document:
        # add metadata
        doc = _enrich_plob_document(index_build_id, plob, doc)
        # return enriched content
        yield doc

def _enrich_plob_document(index_build_id: str,
                          plob: Plob,
                          document: Document
                         ) -> Document:
    """
    Enrich a document (part of a plob) with metadata.

    Args:
        index_build_id: The index build ID to identify this run.
        plob: The (parent) plob to which the document belongs.
        document: The document to be enriched.
    """

    # extract metadata from (parent) document
    plob_id = plob.id
    document_source = plob.metadata.get('source', None)
    if document_source is None:
        document_source = plob.url
    document_title = plob.metadata.get('title', None)
    if document_title is None:
        document_title = plob.metadata.get('title', None)
        if document_title is None:
            document_title = plob.metadata.get('name', None)
            if document_title is None:
                document_title = plob_id

    # extract anker from metadata
    anker = None
    if "anker" in document.metadata:
        anker = document.metadata['anker']
    elif "page_number" in document.metadata:
        anker = f"page-{document.metadata['page_number']}"
    elif "start_index" in document.metadata:
        anker = document.metadata['start_index']

    # add metadata
    document.metadata["index_build_id"] = index_build_id
    document.metadata["plob_id"] = plob_id
    #documentpart_content.metadata["document_source"] = document_source
    if anker:
        document.metadata["source"] = f"{document_source}#{anker}"
    else:
        document.metadata["source"] = document_source
    document.metadata["title"] = plob_id

    return document
