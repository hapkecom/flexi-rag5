from typing import TYPE_CHECKING
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)
import json
from datetime import datetime, timezone
if TYPE_CHECKING:
    from _typeshed.dbapi import DBAPIConnection, DBAPICursor
else:
    DBAPIConnection = any
    DBAPICursor = any
from langchain_core.vectorstores import VectorStore
from common.service.configloader import deep_get, settings
from factory.vectorstore_factory import get_vectorstore

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging
from common.utils.hash_util import sha256sum_str
from common.utils.string_util import str_limit
from index_builder_basics.document_storage_sql_database import get_sql_database_connection_after_setup, get_2nd_sql_database_connection_after_setup
from index_builder_basics.embeddings_cache import get_or_caclulate_and_save_text_sha256_and_embedding_with_sqldb
from model.plob import Plob

logger = logging.getLogger(__name__)

#_sqlCon: DBAPIConnection | None = None
_sqlCon = None



#
# processing multiple documents
#

# iterate over the documents, save them in SQL DB, and add IDs
#def save_docs_in_sqldb(docs_list: Iterator[Document]) -> Iterator[Document]:
#    logger.info(f"save_docs_in_sqldb ...")
#    
#    sqlCon = get_sql_database_connection_after_setup()
#
#    # insert documents (a document represents a a full file/document)
#    num = 0
#    for doc in docs_list:
#        ...
#        yield doc


#
# processing a single document and its content parts
#
def save_single_plob_and_its_documents_in_databases(plob: Plob,
                                                    doc_contents: Iterator[Document]
                                                   ) -> Tuple[Document, Iterator[Document]]:
    """
    Save a single document and its parts (contents)in the SQL DB and the vectorstore.

    NOT LAZY: The document and its parts are processed and saved in the SQL DB and the vectorstore.
    """

    sqlConnection = get_sql_database_connection_after_setup()
    sqlConnection4Embeddings = get_2nd_sql_database_connection_after_setup()
    try:
        # Cleanup: delete old document entry from SQL DB
        delete_old_plob_from_sqldb(sqlConnection, plob.url)

        logger.debug(f"url={plob.url} - deleted old plob entry from SQL DB")

        # Save plob in SQL DB
        now_timestamp = datetime.now(timezone.utc).isoformat()
        plob_stored = save_plob_only_in_sqldb(sqlConnection, plob, now_timestamp)
        plob_id = plob_stored.id
        logger.debug(f"url={plob.url} - saved plob in SQL DB: plob_id={plob_id}")

        # Save documents of plob in SQL DB and vectorstore
        doc_contents_list = list(doc_contents)
        plob_documents_stored = save_documents_of_plob_in_vectorstore_and_sqldb(sqlConnection, sqlConnection4Embeddings, plob_stored, doc_contents_list, now_timestamp)
        # Un-lazy
        plob_documents_stored_done = list(plob_documents_stored)
        logger.debug(f"url={plob.url} - saved doc parts in SQL DB and vectorstore: plob_id={plob_id}, doc_contents_stored={str_limit(plob_documents_stored_done, 1024)}")

        # Done
        sqlConnection.commit()
        logger.debug(f"url={plob.url} - DONE - Saved plob and doc parts in SQL DB and vectorstore: plob_id={plob_id} with {len(plob_documents_stored_done)} doc parts")
        return plob_stored, plob_documents_stored_done

    except Exception as e:
        logger.warning(f"url={plob.url}: {e}")
        try:
            sqlConnection.rollback()
        except Exception as e2:
            logger.warning(f"url={plob.url}: after exception {e}: rollback failed: {e2}")
        raise e
        #return doc, doc_contents
    # Never close sqlConnection
    # finally:
    #    # Close the connection
    #    sqlConnection.close()


# delete olg plob entries from SQL DB
def delete_old_plob_from_sqldb(sqlConnection: DBAPIConnection, plob_url: str):
    # Is the url already in the DB? Then delete related entries now.
    #
    # Attention: The order of deletion is important!

    # Delete in the table "document_content" first
    cursor = sqlConnection.cursor()
    cursor.execute("DELETE FROM plob_document WHERE plob_id IN (SELECT id FROM plob WHERE url=?)", (plob_url,))
    rowcount = cursor.rowcount
    cursor.close()
    logger.debug(f"Deleted {rowcount} row(s) for url={plob_url} from 'plob_document' table")
    
    # Delete in the table "document" second
    cursor = sqlConnection.cursor()
    cursor.execute("DELETE FROM plob WHERE url=?", (plob_url,))
    rowcount = cursor.rowcount
    cursor.close()
    logger.debug(f"Deleted {rowcount} row(s) with url={plob_url} from 'plob' table")


# save plob (without its documents) in SQL DB, and add IDs
def save_plob_only_in_sqldb(sqlConnection: DBAPIConnection, plob: Plob, now_timestamp: str) -> Plob:
    # get id and more
    id = plob.id
    logger.debug(f"plob.id={id}, plob.url={plob.url}, plob.metadata={plob.metadata}")

    # Insert new row into table "document"
    sqlConnection.execute(
        """INSERT INTO plob (id, url, media_type, file_path, file_size, file_sha256, file_last_modified, row_last_modified)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (id, plob.url, plob.media_type, plob.file_path, plob.file_size, plob.file_sha256, plob.file_last_modified, now_timestamp)
    )

    # done
    logger.debug(f"Inserted into 'plob' table row with plob.id={id}, plob.url={plob.url}, plob.metadata={plob.metadata}")
    return plob


# iterate over the documents, save them in SQL DB, and add IDs
def save_documents_of_plob_in_vectorstore_and_sqldb(sqlConnection: DBAPIConnection,
                                                    sqlConnection4Embeddings: DBAPIConnection,
                                                    plob: Plob,
                                                    documents: Iterator[Document],
                                                    now_timestamp: str
                                                   ) -> Iterator[Document]:
    plob_id = plob.id
    logger.debug(f"Start with documents of plob.id={plob_id}, plob.url={plob.url} ...")
    content_count = 0

    # save content parts
    for document in documents:
        logger.debug(f"document.metadata={str_limit(document.metadata, 1024)} document.page_content='{str_limit(document.page_content)}'")

        #
        # save document in vectorstore and SQL DB (if not already there)
        #
        document_sha256 = save_single_document_in_vectorstore_and_sqldb(sqlConnection4Embeddings, document)

        #
        # insert new content part into SQL DB
        #
        document_anker = document.metadata.get("anker")
        logger.debug(f"insert plob_document row: document_id={plob_id}, content_sha256={document_sha256}, document_anker={document_anker}")
        cursor = sqlConnection.cursor()
        cursor.execute(
            """INSERT INTO plob_document (plob_id, document_sha256, document_anker, row_last_modified)
               VALUES (?, ?, ?, ?)""",
            (plob_id, document_sha256, document_anker, now_timestamp)
        )
        cursor.close()
        content_count += 1
        document.metadata["plob_id"] = plob_id
        document.metadata["document_sha256"] = document_sha256
        yield document

    # loop done without exception

    # iteration done
    logger.debug(f"{content_count} plob_document row(s) inserted - DONE")



#
# processing a single content part
#

def save_single_document_in_vectorstore_and_sqldb(sqlConnection4Embeddings: DBAPIConnection,
                                                  document: Document
                                                  ) -> str:
    """
    Add a single document of a single plog to the SQL DB and the vectorstore.

    If the document is already in the SQL DB/vectorstore, nothing will be done.

    Returns: The sha256 hash of the document, or raise an exception in the case of an error
    """

    document_content = None
    document_sha256 = None
    try:
        # preparation
        _vectorStore = get_vectorstore()

        # get/caclulate/save embedding from/to SQL DB
        document_content = document.page_content
        logger.debug("1/4: Before get_or_caclulate_and_save_content_sha256_and_embedding_with_sqldb()")
        document_sha256, content_embedding = get_or_caclulate_and_save_content_sha256_and_embedding_with_sqldb(sqlConnection4Embeddings, document_content)
        logger.debug("2/4: After get_or_caclulate_and_save_content_sha256_and_embedding_with_sqldb()")

        # enrich content metadata before adding it to the vectorstore
        document.metadata["document_sha256"] = document_sha256

        # save content in vectorstore - get/caclulate/save embedding again inclusive SQL DB
        logger.debug(f"3/4: Add document with sha256={document_sha256} to vectorStore")
        resultIds = _vectorStore.add_texts(texts=[document_content], metadatas=[document.metadata],) # ids=[<UUID>])
        logger.debug("4/4: After adding to vectorstore")
        
        # final logging
        if logger.isEnabledFor(logging.DEBUG):
            m = document.metadata
            doc_metadata_str = str_limit(f"{{'source': '{m['source']}', 'title': '{m['title']}', 'part': '{m['part']}', 'part_index': '{m['part_index']}', 'size': '{m['size']}', 'sha256': '{m['sha256']}'}}", 1024)
            logger.debug(f"Added document to vectorStore with id(s)={resultIds}: {doc_metadata_str}, content_sha256={document_sha256}, content_content={str_limit(document_content)}")

        # done
        return document_sha256

    except Exception as e:
        logger.warning(f"content_sha256={document_sha256}, content_content={str_limit(document_content)}): {e}")
        raise e
        #return None


def get_or_caclulate_and_save_content_sha256_and_embedding_with_sqldb(sqlConnection4Embeddings: DBAPIConnection,
                                                                      content: str
                                                                     ) -> Tuple[str, List[float]]:
    """
    Get the embedding of a content part from the SQL DB or calculate and save it SQL DB.

    Returns: The sha256 hash and embedding of the content.
    """
    return get_or_caclulate_and_save_text_sha256_and_embedding_with_sqldb(content, sqlConnection4Embeddings)
