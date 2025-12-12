from utilities.milvus import MilvusProvider
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Milvus Provider...")
    milvus = MilvusProvider()
    
    logger.info("Building indices...")
    try:
        milvus.create_indices()
        logger.info("Index build completed successfully.")
    except Exception as e:
        logger.error(f"Failed to build indices: {e}")

if __name__ == "__main__":
    main()
