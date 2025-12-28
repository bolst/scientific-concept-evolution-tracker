import time
import pandas as pd
import os
from datetime import datetime
from pipeline import SCETPipeline
import utilities.aggregate as agg
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "doc", "report", "images")
os.makedirs(image_dir, exist_ok=True)
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
os.makedirs(results_dir, exist_ok=True)

now = datetime.now().strftime("%Y%m%d_%H%M%S")


def benchmark_pipeline():
    logger.info("Initializing Pipeline...")
    pipeline = SCETPipeline()
    
    queries = [
        "Transformer Attention",
        "Deep Learning",
        "CRISPR",
        "Dark Matter",
        "Quantum Computing",
        "Climate Change",
        "Generative Adversarial Networks",
        "Reinforcement Learning",
        "Graph Neural Networks",
        "Self-Supervised Learning"
    ]
    
    results = []
    
    logger.info("Starting Pipeline Latency Benchmark...")
    
    for query in queries:
        logger.info(f"Processing query: {query}")
        
        # 1. Search
        start_time = time.time()
        df = pipeline.search.search_papers(query, limit=1000, alpha=0.5)
        search_time = time.time() - start_time
        
        if df.empty:
            logger.warning(f"No results for {query}")
            continue
            
        # 2. Clustering
        start_time = time.time()
        concepts, labels, n_clusters_used = pipeline.cluster.get_clustered_concepts(df, n_clusters=None)
        cluster_time = time.time() - start_time
        
        # 3. Era Detection
        start_time = time.time()
        concept_eras = agg.identify_concept_eras(concepts, max_eras=5)
        era_time = time.time() - start_time
        
        # 4. Pivotal Papers
        start_time = time.time()
        pivotals = agg.get_pivotal_papers(concepts, concept_eras, k=3)
        pivotal_time = time.time() - start_time
        
        total_time = search_time + cluster_time + era_time + pivotal_time
        
        results.append({
            "query": query,
            "n_papers": len(df),
            "n_clusters": n_clusters_used,
            "search_time_sec": search_time,
            "cluster_time_sec": cluster_time,
            "era_detection_time_sec": era_time,
            "pivotal_papers_time_sec": pivotal_time,
            "total_time_sec": total_time,
            "param_limit": 1000,
            "param_alpha": 0.5,
            "param_n_clusters": "Auto",
            "param_max_eras": 5,
            "param_pivotal_k": 3
        })
        
    output_file = os.path.join(results_dir, f"pipeline/latency_{now}.csv")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    logger.info(f"Benchmark complete. Results saved to {output_file}")


def main():
    benchmark_pipeline()


if __name__ == "__main__":
    main()
