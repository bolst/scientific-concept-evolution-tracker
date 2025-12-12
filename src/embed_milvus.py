from utilities.embeddings import EmbeddingGenerator
from utilities.metadata import MetadataProvider
from utilities.milvus import MilvusProvider
from tqdm import tqdm
import argparse


def main(shard_id: int, num_shards: int, batch_size: int):
    meta = MetadataProvider()
    embedder = EmbeddingGenerator()
    milvus = MilvusProvider()
    
    print(f"EmbeddingGenerator device: {embedder.device}")
    
    total_papers = meta.get_pending_paper_count(shard_id=shard_id, num_shards=num_shards)
    print(f"Processing {total_papers} papers (Shard {shard_id}/{num_shards})...")
    
    print("=" * 50)

    with tqdm(total=total_papers, unit="papers") as pbar:
        for batch_df in meta.get_pending_papers_batch(batch_size=batch_size, shard_id=shard_id, num_shards=num_shards):
            # try skipping this? we don't want to load entire collection while inserting
            '''
            pending_ids = milvus.filter_existing_ids(batch_df['arxiv_id'].tolist())
            if not pending_ids:
                pbar.update(len(batch_df))
                continue
            df_filtered = batch_df[batch_df['arxiv_id'].isin(pending_ids)].copy()
            '''
            
            embedding = embedder.generate_embeddings(batch_df)
            milvus.prepare_and_ingest(embedding)
            pbar.update(len(batch_df))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed papers and ingest into Milvus")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard ID for parallelism (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size per iteration")
    args = parser.parse_args()

    main(args.shard_id, args.num_shards, args.batch_size)