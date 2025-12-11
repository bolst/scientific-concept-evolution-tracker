import os
import pandas as pd
from pandas import Series
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class MetadataProvider:
    def __init__(self):
        load_dotenv()
        
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        host = os.environ["POSTGRES_HOST"]
        port = os.environ["POSTGRES_PORT"]
        dbname = os.environ["POSTGRES_DB"]
        
        self.database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.database_url)
        logger.info(f"Connected to PostgreSQL at {host}:{port}/{dbname}")
        
    def get_pending_paper_count(self, shard_id: int = 0, num_shards: int = 1) -> int:
        """
        Gets number of pending (unembedded) papers in the database.
        """
        sql = f"""
        SELECT count(*) FROM papers 
        WHERE processed_status = 'pending'
        AND ABS(hashtext(arxiv_id)) % {num_shards} = {shard_id}
        """
        with self.engine.connect() as connection:
            return int(connection.execute(text(sql)).scalar())

    def get_pending_papers_batch(self, batch_size: int = 1000, shard_id: int = 0, num_shards: int = 1):
        """
        Generator that yields batches of pending (unembedded) papers from the database.
        """
        sql = f"""
        SELECT
            arxiv_id,
            title,
            abstract,
            primary_category,
            published_date,
            doi
        FROM
            papers
        WHERE
            processed_status = 'pending'
            AND ABS(hashtext(arxiv_id)) % {num_shards} = {shard_id}
        """
        with self.engine.connect().execution_options(stream_results=True) as connection:
            for chunk in pd.read_sql(text(sql), connection, chunksize=batch_size):
                yield chunk
                
    def get_papers_with_ids(self, arxiv_ids: Series):
        ids_str = f"({ ", ".join("'" + arxiv_ids + "'") })"
        sql = f"""
        SELECT arxiv_id, title, abstract, primary_category, published_date
        FROM papers WHERE arxiv_id IN {ids_str}
        """
        with self.engine.connect() as connection:
            return connection.execute(text(sql)).fetchall()