from .embeddings import EmbeddingGenerator
from .milvus import MilvusProvider
from .metadata import MetadataProvider
import pandas as pd
from pymilvus import AnnSearchRequest, WeightedRanker


class HybridSearch:
    
    def __init__(
        self, 
        metadata: MetadataProvider, 
        milvus: MilvusProvider, 
        embedder: EmbeddingGenerator
        ):
        self.metadata = metadata
        self.milvus = milvus
        self.embedder = embedder
        

    def search_papers(
        self,
        query: str, 
        start_year: int = 1980, 
        end_year: int = 2025, 
        limit: int = 10, 
        alpha: float = 0.5
        ) -> pd.DataFrame:
        # embed query
        dense_vec = self.embedder.dense_model.encode([query], normalize_embeddings=True)[0].tolist()
        sparse_vec = self.embedder.generate_sparse_embedding(query)
        
        # create search requests
        # Fetch more candidates for individual searches
        candidate_limit = min(limit * 10, 16384) # 16384 is milvus limit
        expr = f"publication_year >= {start_year} && publication_year <= {end_year}"
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=candidate_limit,
            expr=expr
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=candidate_limit,
            expr=expr
        )
        
        # perform search
        self.milvus.collection.load()
        res = self.milvus.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=WeightedRanker(alpha, 1-alpha), # alpha * Dense + (1-alpha) * Sparse
            limit=limit,
            output_fields=["arxiv_id", "publication_year"]
        )
        if not res:
            return pd.DataFrame()
        
        # format results
        results = [{
            "arxiv_id": hit.entity.get("arxiv_id"), 
            "score": hit.score, 
            "year": hit.entity.get("publication_year")
            } for hit in res[0]]
        results_df = pd.DataFrame(results)
        if results_df.empty:
            return results_df

        # join Milvus results with Postgres so they contain metadata
        metadata = pd.DataFrame(self.metadata.get_papers_with_ids(results_df['arxiv_id']))
        final_df = results_df.merge(metadata, on='arxiv_id')
        return final_df
