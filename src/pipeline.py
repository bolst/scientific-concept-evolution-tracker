from utilities.embeddings import EmbeddingGenerator
from utilities.metadata import MetadataProvider
from utilities.milvus import MilvusProvider

from utilities.search import HybridSearch
from utilities.cluster import PaperCluster
import utilities.aggregate as agg

import pandas as pd
from dataclasses import dataclass

@dataclass
class PipelineResult:
    query: str
    start_year: int
    end_year: int
    n_clusters: int
    max_eras: int
    pivotal_papers_per_era: int
    
    search_results: pd.DataFrame
    concept_clusters: pd.DataFrame
    concept_labels: dict[int, str]
    concept_eras: dict[str, list[agg.EraStat]]
    pivotal_papers: dict[str, list[agg.EraPivotalPapers]]


class SCETPipeline:
    
    def __init__(self):
        self.metadata = MetadataProvider()
        self.milvus = MilvusProvider(collection_name='arxiv_embeddings_test')
        self.embedder = EmbeddingGenerator()
        
        self.search = HybridSearch(self.metadata, self.milvus, self.embedder)
        self.cluster = PaperCluster(self.milvus, self.embedder)
        
    def process(
        self, 
        query: str, 
        alpha: float = 0.5, 
        start_year: int = 1980, 
        end_year: int = 2025,
        n_clusters: int | None = None,
        max_eras: int = 5,
        pivotal_papers_per_era: int = 3
        ) -> PipelineResult:
        
        df = self.search.search_papers(query, limit=1000, alpha=alpha, start_year=start_year, end_year=end_year)
        concepts, labels, n_clusters_used = self.cluster.get_clustered_concepts(df, n_clusters=n_clusters)
        concept_eras = agg.identify_concept_eras(concepts, max_eras=max_eras)
        pivotals = agg.get_pivotal_papers(concepts, concept_eras, k=pivotal_papers_per_era)
        
        return PipelineResult(
            query = query,
            start_year = start_year,
            end_year = end_year,
            n_clusters = n_clusters_used,
            max_eras = max_eras,
            pivotal_papers_per_era = pivotal_papers_per_era,
            search_results = df,
            concept_clusters = concepts,
            concept_labels = labels,
            concept_eras = concept_eras,
            pivotal_papers = pivotals
        )