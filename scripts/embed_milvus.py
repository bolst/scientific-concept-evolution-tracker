from utilities.embeddings import EmbeddingGenerator
from utilities.metadata import MetadataProvider
from utilities.milvus import MilvusProvider
from tqdm import tqdm


def main():
    meta = MetadataProvider()
    embedder = EmbeddingGenerator()
    milvus = MilvusProvider()
    
    total_papers = meta.get_pending_paper_count()
    print(f"Processing {total_papers} papers...")
    print("=" * 50)

    with tqdm(total=total_papers, unit="papers") as pbar:
        for df in meta.get_pending_papers_batch(batch_size=100):
            # filter out papers that are already embedded
            pending_ids = milvus.filter_existing_ids(df['arxiv_id'].tolist())
            if pending_ids and len(pending_ids) > 0:
                df_filtered = df[df['arxiv_id'].isin(pending_ids)].copy()
                embedding = embedder.generate_embeddings(df_filtered)
                milvus.prepare_and_ingest(embedding)
            # update progress bar
            pbar.update(len(df))
    
    
if __name__ == '__main__':
    main()