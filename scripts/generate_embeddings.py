import os

# Set environment variables to prevent thread contention in multiprocessing
# This must be done before importing libraries that use OpenMP/MKL (like numpy/torch)
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import warnings
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import _src
from scet.core.db import SessionLocal
from scet.core.models import Paper

from dotenv import load_dotenv
load_dotenv()

DEFAULT_BATCH_SIZE = 1000

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "scientific_concepts")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/allenai-specter")
DIMENSION = 768  # SPECTER2 base dimension



def connect_milvus():
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Connected.")

def create_collection_if_not_exists():
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} already exists.")
        return Collection(COLLECTION_NAME)
    
    print(f"Creating collection {COLLECTION_NAME}...")
    
    fields = [
        FieldSchema(name="vector_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="arxiv_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_index", dtype=DataType.INT16),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="year", dtype=DataType.INT16),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
        FieldSchema(name="category_tag", dtype=DataType.INT64),
        FieldSchema(name="citation_count", dtype=DataType.INT32),
    ]
    
    schema = CollectionSchema(fields, "Scientific Concept Evolution Tracker Embeddings")
    
    collection = Collection(COLLECTION_NAME, schema)
    
    # create index
    index_params = {
        "metric_type": "IP", # inner product for normalized embeddings (Cosine similarity)
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Collection created and indexed.")
    return collection

def load_model():
    print(f"loading SPECTER2 model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model

def process_batch(batch_papers, collection, model, db, pool=None):
    if not batch_papers:
        return

    # Identify which papers actually need processing
    # We check Milvus for any paper that is marked as 'embedded' to verify it exists.
    # Papers not marked 'embedded' definitely need processing
    ids_to_check = [p.arxiv_id for p in batch_papers if p.processed_status == 'embedded']
    papers_to_process = [p for p in batch_papers if p.processed_status != 'embedded']
    
    # ensure that papers marked as "embedded" are actually embedded
    if ids_to_check:
        # query Milvus to see which of these IDs already exist
        # this should be fine for batch sizes ~2000
        if (batch_size := len(batch_papers)) > 2000:
            warnings.warn(f"Batch size is {batch_size} > 2000... this may cause performance issues with Milvus query")
        try:
            formatted_ids = [f'"{bid}"' for bid in ids_to_check]
            expr = f"arxiv_id in [{', '.join(formatted_ids)}]"
            res = collection.query(expr, output_fields=["arxiv_id"])
            found_ids = {hit['arxiv_id'] for hit in res}
            
            # if a paper was marked embedded but is not in Milvus, add to process list
            for p in batch_papers:
                if p.processed_status == 'embedded' and p.arxiv_id not in found_ids:
                    papers_to_process.append(p)
        except Exception as e:
            # if check fails, assume we need to process to be safe
            print(f"Error checking Milvus for batch: {e}")
            for p in batch_papers:
                if p.processed_status == 'embedded':
                    papers_to_process.append(p)

    # if none to process, exit early
    if not papers_to_process or len(papers_to_process) == 0:
        return

    # generate embeddings for the filtered list
    texts = []
    for p in papers_to_process:
        title = p.title or ""
        abstract = p.abstract or ""
        sep_token = getattr(model.tokenizer, 'sep_token', ' ') or ' '
        text = title + sep_token + abstract 
        texts.append(text)
    
    embeddings = model.encode(
        texts, 
        pool=pool,
        convert_to_tensor=False, 
        normalize_embeddings=True
    )

    # prepare data for Milvus
    mr_arxiv_ids = []
    mr_chunk_indices = []
    mr_embeddings = []
    mr_years = []
    mr_timestamps = []
    mr_category_tags = []
    mr_citation_counts = []
    
    for j, p in enumerate(papers_to_process):
        mr_arxiv_ids.append(p.arxiv_id)
        mr_chunk_indices.append(0) # in case I want to encode full paper later on
        # Ensure embedding is a list for Milvus
        emb = embeddings[j]
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        mr_embeddings.append(emb)
        mr_years.append(p.published_date.year if p.published_date else 2000)
        mr_timestamps.append(int(time.mktime(p.published_date.timetuple())) if p.published_date else 0)
        # simple hash for category
        mr_category_tags.append(hash(p.primary_category) % 10000 
                                if p.primary_category 
                                else 0)
        mr_citation_counts.append(0) # TODO: may not even need this

    data = [
        mr_arxiv_ids, mr_chunk_indices, mr_embeddings, mr_years, 
        mr_timestamps, mr_category_tags, mr_citation_counts
    ]
    
    # upsert to Milvus
    try:
        formatted_ids = [f'"{bid}"' for bid in mr_arxiv_ids]
        expr = f"arxiv_id in [{', '.join(formatted_ids)}]"
        collection.delete(expr)
    except Exception as e:
        # delete might fail if doesn't exist... which is fine
        print(f"deletion failed: {e}")
        pass
    collection.insert(data)
    
    # update DB
    for p in papers_to_process:
        p.processed_status = "embedded"

def generate_embeddings(batch_size=DEFAULT_BATCH_SIZE, num_workers=1):
    connect_milvus()
    collection = create_collection_if_not_exists()
    collection.load() 
    model = load_model()
    db = SessionLocal()
    
    # initialize multiprocessing pool if requested
    pool = None
    if num_workers > 1:
        print(f"Starting {num_workers} worker processes for embedding generation...")
        pool = model.start_multi_process_pool(target_devices=['cpu'] * num_workers)
    
    try:
        # for progress bar
        total_papers = db.query(Paper).count()
        print(f"Total papers in DB: {total_papers}")
        
        pbar = tqdm(total=total_papers)
        
        last_seen_id = None
        base_query = db.query(Paper).order_by(Paper.arxiv_id)
        while True:
            # query paper batch
            query = base_query
            if last_seen_id:
                query = base_query.filter(Paper.arxiv_id > last_seen_id)
            batch_papers = query.limit(batch_size).all()
            
            # if none, then we have processed all and we are done
            if not batch_papers:
                break
            
            # Update for next iteration
            last_seen_id = batch_papers[-1].arxiv_id
                
            # Process and commit each batch
            process_batch(batch_papers, collection, model, db, pool)
            db.commit()
            
            pbar.update(len(batch_papers))
            
        pbar.close()
            
        print("Done generating and inserting embeddings")
        collection.flush()
        print(f"Collection row count: {collection.num_entities}")
        
    finally:
        if pool:
            print("Stopping worker pool...")
            model.stop_multi_process_pool(pool)
        db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for papers.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for processing papers")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes for embedding generation")
    args = parser.parse_args()
    
    print(f"Generating embeddings with batch_size={args.batch_size}, num_workers={args.num_workers}")
    generate_embeddings(args.batch_size, args.num_workers)
