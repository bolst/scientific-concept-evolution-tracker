from utilities.embeddings import EmbeddingGenerator
from utilities.metadata import MetadataProvider
from utilities.milvus import MilvusProvider

import logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s][%(name)s] %(message)s')

from utilities.search import HybridSearch
from utilities.cluster import PaperCluster
import utilities.aggregate as agg

meta = MetadataProvider()
milvus = MilvusProvider(collection_name='arxiv_embeddings_test')
embedder = EmbeddingGenerator()

hs = HybridSearch(meta, milvus, embedder)
df = hs.search_papers('Transformer', limit=1000)
print(f"Returned dataframe of shape {df.shape}")
print(df.head())

cl = PaperCluster(milvus, embedder)
concepts, labels, n_clusters = cl.get_clustered_concepts(df)
print("\nIdentified Concepts:")
for cid, label in labels.items():
    count = len(concepts[concepts['cluster'] == cid])
    print(f"Cluster {cid} ({count} papers): {label}")
print(concepts[['year', 'title', 'cluster_label']].head(10))

concept_eras = agg.identify_concept_eras(concepts)
print("Identified Concept Eras:")
for label, era_list in concept_eras.items():
    print(f"\nCONCEPT: {label}")
    for era in era_list:
        print(f"  - {era.start} to {era.end}: ~{era.avg_count:.1f} papers/yr")

pivotals = agg.get_pivotal_papers(concepts, concept_eras)
print("Pivotal Papers per Era")
for label, eras in pivotals.items():
    print(f"\nCONCEPT: {label}")
    for era in eras:
        print(f"\t[{era.period}]")
        for p in era.papers:
            print(f"\t - ({p.year}|{p.score:.4f}) {p.title}")