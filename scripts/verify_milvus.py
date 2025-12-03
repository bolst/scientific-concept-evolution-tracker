import os
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "scientific_concepts")

def verify_milvus_data():
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("Connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist.")
        return

    collection = Collection(COLLECTION_NAME)
    
    # Load collection to memory to query
    print(f"Loading collection '{COLLECTION_NAME}'...")
    collection.load()
    
    num_entities = collection.num_entities
    print(f"Number of entities in collection: {num_entities}")
    
    if num_entities > 0:
        print("\nSample vector retrieval (limit 5):")
        results = collection.query(
            expr="", 
            output_fields=["arxiv_id", "chunk_index", "year", "citation_count"],
            limit=5
        )
        for res in results:
            print(res)
    else:
        print("Collection is empty.")

if __name__ == "__main__":
    verify_milvus_data()
