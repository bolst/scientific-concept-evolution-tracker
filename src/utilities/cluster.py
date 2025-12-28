from .milvus import MilvusProvider, hash_to_int64
from .embeddings import EmbeddingGenerator
from sklearn.metrics import silhouette_score
from warnings import warn
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PaperCluster:
    
    def __init__(self, milvus: MilvusProvider, embedder: EmbeddingGenerator):
        self.milvus = milvus
        self.embedder = embedder

    def get_optimal_cluster_count(self, arxiv_ids: list[str], max_k: int = 8):
        if len(arxiv_ids) < 2:
            return 2
        
        paper_ids = list(map(hash_to_int64, arxiv_ids))
        res = self.milvus.collection.query(
            expr=f"paper_id in {paper_ids}",
            output_fields=["dense_vector"]
        )
        if not res:
            return 2
            
        vectors = [x['dense_vector'] for x in res]
        matrix = np.stack(vectors)
        
        scores = []
        ks = range(2, min(max_k + 1, len(matrix)))
        logger.info(f"Evaluating cluster counts from {ks[0]} to {ks[-1]}...")
        logger.debug("Silhouette Scores:")
        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(matrix)
            score = silhouette_score(matrix, labels)
            scores.append(score)
            logger.debug(f"\tk={k}: {score:.4f}")
            
        best_k = ks[np.argmax(scores)]
        return best_k



    def get_clustered_concepts(
        self,
        search_results_df: pd.DataFrame,
        n_clusters: int | None = None
        ) -> tuple[pd.DataFrame, dict, int]:
        """
        Clusters search results into sub-concepts and tracks them over time.
        Returns tuple of [cluster dataframe, cluster labels {clusterid:label}, number clusters used]
        """
        
        if search_results_df.empty:
            warn("search_results_df is empty, returning empty tuple")
            return pd.DataFrame(), {}, n_clusters
        
        if n_clusters is None or n_clusters < 2:
            n_clusters = self.get_optimal_cluster_count(search_results_df['arxiv_id'].tolist())
            logger.info(f"Determined {n_clusters} to be number of optimal clusters")

        # fetch vectors from Milvus for the given papers
        paper_ids = search_results_df['arxiv_id'].apply(hash_to_int64).tolist()
        res = self.milvus.collection.query(
            expr=f"paper_id in {paper_ids}",
            output_fields=["arxiv_id", "dense_vector", "sparse_vector"]
        )
        vectors_df = pd.DataFrame(res)
        
        # merge Milvus results for vectors
        # search_results_df already has metadata (title, etc.) from search_papers
        df = search_results_df.merge(vectors_df, on='arxiv_id')
        
        if len(df) < n_clusters:
            logger.warning(f"Not enough data points ({len(df)}) for {n_clusters} clusters.")
            return df, {}, n_clusters

        # build clusters from dense vectors
        logger.info(f"Clustering {len(df)} papers into {n_clusters} concepts...")
        matrix = np.stack(df['dense_vector'].values)
        
        # scikit recommends using MiniBatchKMeans if > 10000 samples
        KMCluster = MiniBatchKMeans if len(df) > 10000 else KMeans
        kmeans = KMCluster(n_clusters=n_clusters, random_state=42, n_init=10)
            
        df['cluster'] = kmeans.fit_predict(matrix)
        
        # build cluster labels with sparse vectors
        cluster_labels = {}
        logger.info("Generating cluster labels...")
        for cluster_id in range(n_clusters):
            cluster = df[df['cluster'] == cluster_id]
            
            # build map (token -> token's total weight in the cluster)
            token_weights = defaultdict(float)
            for sparse_vec in cluster['sparse_vector']:
                for token_id, weight in sparse_vec.items():
                    token_weights[token_id] += weight
                    
            # decode top tokens (i.e., tokens with largest weight)
            top_tokens = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            decoded_tokens = [self.embedder.sparse_tokenizer.decode([token]) for token, _ in top_tokens]
            
            # sometimes cluster labels are generated as tokens that are
            # obviously incorrectly parsed (e.g., ##er or ##ing), so just filter them out for now
            # TODO: fix clustering
            decoded_tokens = [d for d in decoded_tokens if '#' not in d]
            # use decoded tokens (i.e., sub-concepts) as labels
            label = ", ".join(decoded_tokens)
            cluster_labels[cluster_id] = label
            
        df['cluster_label'] = df['cluster'].map(cluster_labels)
        return df, cluster_labels, n_clusters

