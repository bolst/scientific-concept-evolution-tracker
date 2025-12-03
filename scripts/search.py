import os, sys
import argparse
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from sqlalchemy.orm import Session

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from scet.db import SessionLocal
from scet.models import Paper

from dotenv import load_dotenv
load_dotenv()


MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "scientific_concepts")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/allenai-specter")


def connect_milvus():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        sys.exit(1)

def get_model():
    print(f"Loading model {EMBEDDING_MODEL}...")
    return SentenceTransformer(EMBEDDING_MODEL)

def search(query: str, year: int = None, limit: int = 10):
    # connect/load
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    model = get_model()
    
    # encode and build query
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    search_params = {
        "metric_type": "IP", 
        "params": {"nprobe": 10}
    }
    expr = f"year == {year}" if year else None
    
    # search
    print(f"\nSearching for '{query}'" + (f" in {year}" if year else "") + "...")
    results = collection.search(
        data=[query_embedding], 
        anns_field="embedding", 
        param=search_params, 
        limit=limit, 
        expr=expr,
        output_fields=["arxiv_id", "year"]
    )
    
    # fetch metadata from DB
    db: Session = SessionLocal()
    hits = results[0]
    
    print(f"\nFound {len(hits)} results:")
    print("-" * 80)
    
    for hit in hits:
        arxiv_id = hit.entity.get("arxiv_id")
        score = hit.distance
        year_val = hit.entity.get("year")
        
        paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
        
        title = paper.title if paper else "Unknown Title"
        # truncate title if too long
        if len(title) > 60:
            title = title[:57] + "..."
            
        print(f"[{year_val}] [{arxiv_id}] Score: {score:.2f} | {title}")
        
    print("-" * 80)
    db.close()

def compare_years(query: str, year1: int, year2: int, limit: int = 3):
    print(f"\n=== Comparing '{query}' between {year1} and {year2} ===")
    search(query, year=year1, limit=limit)
    search(query, year=year2, limit=limit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search scientific papers.")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--year", type=int, help="Filter by year", default=None)
    parser.add_argument("--limit", type=int, help="Number of results", default=10)
    parser.add_argument("--compare", type=int, help="Compare with this year (requires --year to be set)", default=None)

    args = parser.parse_args()
    
    if args.compare and args.year:
        compare_years(args.query, args.year, args.compare, args.limit)
    else:
        search(args.query, args.year, args.limit)
