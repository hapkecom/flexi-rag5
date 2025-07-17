from typing import TYPE_CHECKING

from functools import cache
from typing import List, Dict, Optional, Tuple
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
import logging
import json
from datetime import datetime, timezone
from common.utils.hash_util import sha256sum_str
from common.utils.string_util import str_limit
from common.service.configloader import deep_get, settings
from factory.llm_factory import get_default_embeddings
from .document_storage_sql_database import get_2nd_sql_database_connection_after_setup
if TYPE_CHECKING:
    from _typeshed.dbapi import DBAPIConnection, DBAPICursor
else:
    DBAPIConnection = any
    DBAPICursor = any

logger = logging.getLogger(__name__)



#
# Embedding model setup
#

@cache
def get_cached_default_embeddings(sqlConnection4Embeddings: DBAPIConnection = get_2nd_sql_database_connection_after_setup()) -> Embeddings:
    """Get the cache/performance-optimized default embedding model.

    Returns:
        The cached default embedding model.
    """

    global defaultSqlConnection4Embeddings
    if sqlConnection4Embeddings is not None:
        defaultSqlConnection4Embeddings = sqlConnection4Embeddings 

    return CachedEmbeddings()

#
# Embedding model wrapper
#

embedding_model_id = deep_get(settings, "config.common.embedding_model_id")

class CachedEmbeddings(BaseModel, Embeddings):
    """Embedding from the configured default embedding model.
       Cached in SQL database table "content".
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings of texts from the SQL DB or calculate and save it SQL DB.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        # iterate over texts
        embeddings = []
        for text in texts:
            embedding = self.embed_document(text)
            embeddings.append(embedding)
        return embeddings

    def embed_document(self, text: str) -> List[float]:
        """Get the embedding of a single text from the SQL DB or calculate and save it SQL DB.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """

        logger.debug(f"embed_document START ...")
        sha256, embedding = get_or_caclulate_and_save_text_sha256_and_embedding_with_sqldb(text)
        logger.debug(f"DONE: embedding_model_id='{embedding_model_id}', sha256={sha256}, embedding_len={len(embedding)}")

        return embedding

    def embed_query(self, text: str) -> List[float]:
        """Get cached embedding.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Get cached embedding (async).

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]


#
# Basic functions
#

# Set global variable sqlConnection4Embeddings
defaultSqlConnection4Embeddings: Optional[DBAPIConnection] = None

def get_or_caclulate_and_save_text_sha256_and_embedding_with_sqldb(
        text: str,
        sqlConnection4Embeddings: Optional[DBAPIConnection] = defaultSqlConnection4Embeddings
        ) -> Tuple[str, List[float]]:
    """
    Get the embedding of a text from the SQL DB or calculate and save it SQL DB.
    Use SQL database table "document" for caching.

    Returns: The sha256 hash and embedding of the content.
    """

    # Preparation
    content_sha256 = sha256sum_str(text)
    #logger.debug(f"content_sha256={content_sha256}, embedding_model_id='{embedding_model_id}')")
    embedding = None
    sqlConnection = sqlConnection4Embeddings or defaultSqlConnection4Embeddings

    # Action
    try:
        # Pre-check DB
        if sqlConnection is None:
            logger.warning("sqlConnection4Embeddings is None - continue without SQL DB")
            embeddings: Embeddings = get_default_embeddings()
            embedding = embeddings.embed_documents([text])[0]
            return content_sha256, embedding
        # Continue with SQL DB

        # Check if content is already in SQL DB
        cursor1 = sqlConnection.cursor()
        cursor1.execute("SELECT embedding_json FROM document WHERE sha256=? AND embedding_model_id=?", (content_sha256,embedding_model_id,))
        row = cursor1.fetchone()

        if row:
            # Document is already in SQL DB: read embedding
            logger.debug(f"embedding of document already in SQL DB: sha256={content_sha256}, content={str_limit(text)}")
            embedding_json = row[0]
            embedding = json.loads(embedding_json)
        else:
            # Document is NOT in SQL DB
            logger.debug(f"embedding of document NOT YET in SQL DB - calculate it: sha256={content_sha256}, content={str_limit(text)}")

            # claculate the embedding
            embeddings: Embeddings = get_default_embeddings()
            embedding = embeddings.embed_documents([text])[0]
            logger.debug(f"calculate embedding of document DONE")

            # save the embedding in the SQL DB
            #logger.debug(f"save embedding of document {content_sha256} in SQL DB: {embedding}")
            embedding_json = json.dumps(embedding)
            cursor2 = sqlConnection.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor2.execute(
                """INSERT INTO document (sha256, content, embedding_model_id, embedding_json, row_last_modified)
                             VALUES (?, ?, ?, ?, ?)""",
                (content_sha256, text, embedding_model_id, embedding_json, now)
            )
            cursor2.close()
            logger.debug(f"inserted document row: sha256={content_sha256}, content={str_limit(text)}, embedding_model_id={embedding_model_id} into SQL DB")

        # DB cleanup
        cursor1.close()
        sqlConnection.commit()

        return content_sha256, embedding

    except Exception as e:
        logger.warning(f"content_sha256={content_sha256}, content={str_limit(text)}): {e}")
        raise e
