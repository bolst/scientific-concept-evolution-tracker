import os, sys
import time
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

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from scet.db import SessionLocal
from scet.models import Paper

from dotenv import load_dotenv
load_dotenv()



BATCH_SIZE = 20 # batch size of papers to generate embeddings for
LIMIT = int(os.getenv("DATASET_LIMIT", 1000))
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

def generate_embeddings():
    connect_milvus()
    collection = create_collection_if_not_exists()
    model = load_model()
    
    db = SessionLocal()
    
    # fetch papers that we have not embedded yet
    papers = db.query(Paper).filter(Paper.processed_status != "embedded").limit(LIMIT).all()
    print(f"Fetched {len(papers)} papers from DB.")
        
    for i in tqdm(range(0, len(papers), BATCH_SIZE)):
        batch_papers = papers[i:i+BATCH_SIZE]
        
        texts = []
        for p in batch_papers:
            title = p.title or ""
            abstract = p.abstract or ""
            sep_token = getattr(model.tokenizer, 'sep_token', ' ') or ' '
            text = title + sep_token + abstract 
            texts.append(text)
            
        embeddings = model.encode(texts, convert_to_tensor=False)

        # prepare data for Milvus
        # Schema: [vector_id, arxiv_id, chunk_index, embedding, year, timestamp, category_tag, citation_count]
        
        mr_arxiv_ids = []
        mr_chunk_indices = []
        mr_embeddings = []
        mr_years = []
        mr_timestamps = []
        mr_category_tags = []
        mr_citation_counts = []
        
        for j, p in enumerate(batch_papers):
            mr_arxiv_ids.append(p.arxiv_id)
            mr_chunk_indices.append(0) # Only 1 chunk for now
            mr_embeddings.append(embeddings[j])
            
            year = p.published_date.year if p.published_date else 2000
            mr_years.append(year)
            
            ts = int(time.mktime(p.published_date.timetuple())) if p.published_date else 0
            mr_timestamps.append(ts)
            
            # Simple hash for category tag
            cat_hash = hash(p.primary_category) % 10000 if p.primary_category else 0
            mr_category_tags.append(cat_hash)
            
            mr_citation_counts.append(0) # just put 0 for now
            
        data = [
            mr_arxiv_ids,
            mr_chunk_indices,
            mr_embeddings,
            mr_years,
            mr_timestamps,
            mr_category_tags,
            mr_citation_counts
        ]
        
        collection.insert(data)
        
        # TODO: set processed_status in db to "embedded"
        
    print("Done generating and inserting embeddings")
    
    # ensure data is persisted
    collection.flush()
    print(f"Collection row count: {collection.num_entities}")

if __name__ == "__main__":
    generate_embeddings()
