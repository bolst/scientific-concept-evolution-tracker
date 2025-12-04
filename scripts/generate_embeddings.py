import os
import time
import warnings
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



BATCH_SIZE = 20 # batch size of papers to generate embeddings for
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

def process_batch(batch_papers, collection, model, db):
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
        # this should be fine for batch sizes ~1000
        if (batch_size := len(batch_papers)) > 1000:
            warnings.warn(f"Batch size is {batch_size} > 1000... this may cause performance issues")
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
    embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

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
        mr_chunk_indices.append(0) 
        mr_embeddings.append(embeddings[j])
        mr_years.append(p.published_date.year if p.published_date else 2000)
        mr_timestamps.append(int(time.mktime(p.published_date.timetuple())) if p.published_date else 0)
        mr_category_tags.append(hash(p.primary_category) % 10000 
                                if p.primary_category 
                                else 0) # simple hash for category
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
        p.processed_status = 'embedded'
    # db.commit()

def chunked_iterable(iterable, size):
    """Yields chunks of an iterable"""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            pass
        if chunk:
            yield chunk
        if len(chunk) < size:
            break

def generate_embeddings():
    connect_milvus()
    collection = create_collection_if_not_exists()
    collection.load() 
    model = load_model()
    db = SessionLocal()
    
    # for progress bar
    total_papers = db.query(Paper).count()
    print(f"Total papers in DB: {total_papers}")
    
    # stream results from DB instead of loading all at once
    paper_stream = db.query(Paper).yield_per(1000)
    
    # Process in batches
    # Use a larger batch size for DB/Milvus checks (e.g. 100)
    # Embedding model batching is handled inside process_batch if needed, 
    # but here we just pass 100 papers to process_batch.
    # SPECTER2 can handle 20-30 easily... 100 might be slow on CPU
    for batch in tqdm(chunked_iterable(paper_stream, BATCH_SIZE), total=total_papers//BATCH_SIZE):
        process_batch(batch, collection, model, db)
    db.commit()
        
    print("Done generating and inserting embeddings")
    collection.flush()
    print(f"Collection row count: {collection.num_entities}")

if __name__ == "__main__":
    generate_embeddings()
