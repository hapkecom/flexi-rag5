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
if TYPE_CHECKING:
    from _typeshed.dbapi import DBAPIConnection, DBAPICursor
else:
    DBAPIConnection = any
    DBAPICursor = any
import shortuuid
from common.service.configloader import deep_get, settings
from factory.sql_database_factory import get_sql_database_connection

import logging
from common.utils.hash_util import sha256sum_str
from common.utils.string_util import str_limit


logger = logging.getLogger(__name__)

#_sqlCon: DBAPIConnection | None = None
_sqlCon = None
_sqlCon2 = None


#
# (Persistent) Data Model:
#
# The basic idea is to have documents (table 'document') and 
# # connected document parts (table 'document_part') in the SQL DB.
# These documents and their connected parts can change over time.
#
# And we have a parts (=text snippets + their sha256 hash + their embedding )
# which are stored/cached almost forever to save embedding-calculation costs and time.
# These parts are stored in the 'part' table in the SQL DB
# and in the vectorstore DB.
# Parts are identified and connected by their sha256 hash.
#
# setup if necessary

# a document represents a a full file/document
DB_TABLE_plob = """CREATE TABLE IF NOT EXISTS plob (
                        id TEXT NOT NULL PRIMARY KEY,
                        url TEXT COMMENT "URL or file path of the document" NOT NULL,
                        media_type TEXT NOT NULL,
                        file_path TEXT,
                        file_size INTEGER,
                        file_sha256 TEXT,
                        file_last_modified TEXT COMMENT "timestamp of the last modification of the file/source",
                        row_last_modified TEXT COMMENT "timestamp of the last database row modification" NOT NULL
                    )"""

# A "content" represents a text and its embedding, e.g. part of a document after splitting,
# e.g., a page, a paragraph, a part of a page.
#
# Entries of "content" are not deleted even if they are not included in the latest version of the vectorstore -
# in this case only "document_content" entries are deleted.
# Furthermore "content" are persistent and a kind of long-term cache,
# to save embedding-calculation costs and time.

DB_TABLE_document =  """CREATE TABLE IF NOT EXISTS document (
                            sha256 TEXT COMMENT "sha256 hash of the document/text, also used as ID here" NOT NULL,
                            content TEXT COMMENT "content/text, e.g. a part of a document" NOT NULL,
                            embedding_model_id TEXT COMMENT "embedding model used to create the embedding of this part",
                            embedding_json TEXT COMMENT "embedding of the content of this part" NOT NULL,
                            row_last_modified TEXT COMMENT "timestamp of the last database row modification" NOT NULL,
                            UNIQUE(sha256, embedding_model_id)
                        )""" # COMMENT "sha256+embedding_model are the primary key"

# Connection between a "plob" and its "document" content parts
DB_TABLE_plob_document = """CREATE TABLE IF NOT EXISTS plob_document (
                                plob_id TEXT NOT NULL,
                                document_sha256 TEXT NOT NULL,
                                document_anker TEXT COMMENT "position of the document in the plob - e.g., page number, paragraph number, ...",
                                row_last_modified TEXT COMMENT "timestamp of the last database row modification" NOT NULL
                            )""" # alias "document_text"

#
# Helper functions
#
# ...?


#
# Database debugging functions
#

def get_all_plobs_from_sqldb() -> List[Dict[str, Any]]:
    # get all rows
    sqlCon = get_sql_database_connection_after_setup()
    cur = sqlCon.cursor()
    cur.execute("SELECT id, url, media_type, file_path, file_size, file_sha256, file_last_modified, row_last_modified FROM plob")
    rows = cur.fetchall()
    cur.close()

    # map rows to document dictionaries
    document_dicts = [{
        "id": row[0],
        "url": row[1],
        "media_type": row[2],
        "file_path": row[3],
        "file_size": row[4],
        "file_sha256": row[5],
        "file_last_modified": row[6],
        "row_last_modified": row[6],
    } for row in rows]

    return document_dicts

def print_all_plobs_from_sqldb():
    logger.info("""All "plob" rows in SQL DB:""")
    docs = get_all_plobs_from_sqldb()
    docs_values = [list(doc.values()) for doc in docs]

    for row in docs_values:
        logger.info("  DB row: "+str_limit(row, 200))
    logger.info("""All "plob" rows in SQL DB - DONE""")


def get_all_plob_documents_from_sqldb() -> List[Dict[str, Any]]:
    # get all rows
    sqlCon = get_sql_database_connection_after_setup()
    cur = sqlCon.cursor()
    cur.execute("SELECT plob_id, document_sha256, document_anker, row_last_modified FROM plob_document")
    rows = cur.fetchall()
    cur.close()

    # map rows to document_part dictionaries
    document_content_dicts = [{
        "plob_id": row[0],
        "document_sha256": row[1],
        "document_anker": row[2],
        "row_last_modified": row[3],
    } for row in rows]

    return document_content_dicts

def print_all_plob_documents_from_sqldb():
    logger.info("""All "plob_document" rows in SQL DB:""")
    doc_contents = get_all_plob_documents_from_sqldb()
    doc_contents_values = [list(doc_part.values()) for doc_part in doc_contents]

    for row in doc_contents_values:
        logger.info("  DB row: "+str_limit(row, 200))
    logger.info("""All "plob_document" rows in SQL DB - DONE""")

def get_all_documents_from_sqldb() -> List[Dict[str, Any]]:
    # get all rows
    sqlCon = get_sql_database_connection_after_setup()
    cur = sqlCon.cursor()
    cur.execute("SELECT sha256, content, embedding_model_id, embedding_json, row_last_modified FROM document")
    rows = cur.fetchall()
    cur.close()
    # map rows to part dictionaries
    content_dicts = [{
        "sha256": row[0],
        "content": row[1],
        "embedding_model_id": row[2],
        "embedding_json_len": len(row[3]),
        "row_last_modified": row[3],
    } for row in rows]

    return content_dicts

def print_all_documents_from_sqldb():
    logger.info("""All "document" rows in SQL DB :""")
    documents = get_all_documents_from_sqldb()
    document_values = [list(document.values()) for document in documents]

    for row in document_values:
        logger.info("  DB row: "+str_limit(row, 200))
    logger.info("""All "document" rows in SQL DB - DONE""")

def print_all_from_sqldb():
    logger.info("== printall_in_sqldb()")

    print_all_plobs_from_sqldb()
    print_all_documents_from_sqldb()
    print_all_plob_documents_from_sqldb()



#
# basic database functions
#

def get_sql_database_connection_after_setup() -> DBAPIConnection:
    """
    Get the (long-term) SQL database connection, setup the tables if necessary.

    Returns: the SQL database connection
    """

    global _sqlCon
    if _sqlCon is None:
        _sqlCon = get_sql_database_connection()
            # see also:
            # - https://docs.python.org/3/library/sqlite3.html#sqlite3.threadsafety
            # - https://discuss.python.org/t/is-sqlite3-threadsafety-the-same-thing-as-sqlite3-threadsafe-from-the-c-library/11463

        # setup tables if necessary
        _sqlCon.execute(DB_TABLE_plob)
        _sqlCon.execute(DB_TABLE_document)
        _sqlCon.execute(DB_TABLE_plob_document)

    return _sqlCon

def get_2nd_sql_database_connection_after_setup() -> DBAPIConnection:
    """
    Get an additional SQL database connection, for short transactions.
    Wetup the tables if necessary.

    Returns: the SQL database connection
    """

    # Setup tables if necessary
    get_sql_database_connection_after_setup() 

    global _sqlCon2
    if _sqlCon2 is None:
        _sqlCon2 = get_sql_database_connection()
            # see also:
            # - https://docs.python.org/3/library/sqlite3.html#sqlite3.threadsafety
            # - https://discuss.python.org/t/is-sqlite3-threadsafety-the-same-thing-as-sqlite3-threadsafe-from-the-c-library/11463

    return _sqlCon2
