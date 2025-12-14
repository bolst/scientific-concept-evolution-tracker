from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from dotenv import load_dotenv
from warnings import warn
import pandas as pd
import os
import hashlib
import logging

logger = logging.getLogger(__name__)


DEFAULT_COLLECTION_NAME = "arxiv_embeddings"


def hash_to_int64(s: str) -> int:
    """
    Generates a stable 64-bit integer hash from a string.
    We use MD5 and take the first 15 hex characters (60 bits) to ensure it fits 
    within a signed 64-bit integer (Milvus Int64 limit).
    """
    return int(hashlib.md5(s.encode()).hexdigest()[:15], 16)

def get_partition_key(year, domain_str) -> int:
    key_str = f"{year}_{domain_str}"
    return hash_to_int64(key_str)


class MilvusProvider:
    
    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME):
        load_dotenv()

        self.milvus_host = os.environ["MILVUS_HOST"]
        self.milvus_port = os.environ["MILVUS_PORT"]
        connections.connect("default", host=self.milvus_host, port=self.milvus_port)
        logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        self.collection = self._get_collection(collection_name)

    
    def _get_collection(self, collection_name: str):
        '''
        The following is an upsert. No duplicates will be created.
        '''
        fields = [
            FieldSchema(name="paper_id", dtype=DataType.INT64, is_primary=True, description="Hashed arXiv ID"),
            FieldSchema(name="arxiv_id", dtype=DataType.VARCHAR, max_length=32, description="Original arXiv ID"),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=768, description="SPECTER2 embedding"),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, description="SPLADE embedding"),
            FieldSchema(name="publication_year", dtype=DataType.INT16, description="Year for filtering"),
            FieldSchema(name="partition_key", dtype=DataType.INT64, is_partition_key=True, description="Hash of Year + Domain"),
        ]
        schema = CollectionSchema(fields, "ArXiv Collection")
        collection = Collection(collection_name, schema)
        return collection

    def create_indices(self):
        """
        Builds indices for the collection. 
        This should be called AFTER bulk ingestion to avoid memory overhead and slowdowns.
        """
        logger.info('\n' + ('=' * 50) + f"\nBuilding indices for collection '{self.collection.name}'\n" + ('=' * 50))
        logger.info("Building Dense Vector Index (HNSW)...")
        self.collection.create_index(field_name="dense_vector", index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        })
        
        logger.info("Building Sparse Vector Index...")
        self.collection.create_index(field_name="sparse_vector", index_params={
            "metric_type": "IP", # Inner Product is standard for sparse
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2}
        })
        
        logger.info("Building Scalar Indices...")
        self.collection.create_index(field_name="publication_year", index_name="year_index")
        self.collection.create_index(field_name="paper_id", index_name="paper_id_index")
        logger.info("All indices successfully built!")
    
    
    # can probably delete this. need to verify first
    def filter_existing_ids(self, arxiv_ids: list[str]) -> list[str]:
        """
        Checks Milvus for existing IDs and returns only the ones that are NOT in the database.
        Requires collection to be loaded.
        """
        if not arxiv_ids:
            return []
            
        # Ensure collection is loaded for query
        try:
            logger.info(f"Loading collection '{self.collection.name}'...")
            self.collection.load()
            logger.info(f"Collection '{self.collection.name}' successfully loaded.")
        except Exception as e:
            logger.warning(f"Failed to load collection for filtering: {e}")
            return arxiv_ids

        hashed_ids = [hash_to_int64(aid) for aid in arxiv_ids]
        res = self.collection.query(
            expr=f"paper_id in {hashed_ids}",
            output_fields=["arxiv_id"]
        )
        existing_ids = set(item['arxiv_id'] for item in res)
        # return IDs NOT in existing set
        return [aid for aid in arxiv_ids if aid not in existing_ids]


    def prepare_and_ingest(self, df: pd.DataFrame):
        insert_data = {
            "paper_id": [],
            "arxiv_id": [],
            "dense_vector": [],
            "sparse_vector": [],
            "publication_year": [],
            "partition_key": []
        }
                
        for _, row in df.iterrows():
            try:
                p_id = hash_to_int64(row['arxiv_id'])
                year = row['published_date'].year
                part_key = get_partition_key(year, row['primary_category'])
                
                insert_data["paper_id"].append(p_id)
                insert_data["arxiv_id"].append(str(row['arxiv_id']))
                insert_data["dense_vector"].append(row['dense_vector'])
                insert_data["sparse_vector"].append(row['sparse_vector'])
                insert_data["publication_year"].append(year)
                insert_data["partition_key"].append(part_key)
            except Exception as e:
                warn(f"Error processing {row['arxiv_id']}: {e}")
                continue
                
        self.collection.upsert([
            insert_data["paper_id"],
            insert_data["arxiv_id"],
            insert_data["dense_vector"],
            insert_data["sparse_vector"],
            insert_data["publication_year"],
            insert_data["partition_key"]
        ])