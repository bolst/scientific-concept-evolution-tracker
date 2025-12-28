import os
import time
import pandas as pd
from datetime import datetime
from pipeline import SCETPipeline
import utilities.aggregate as agg
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "doc", "report", "images")
os.makedirs(image_dir, exist_ok=True)
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
os.makedirs(results_dir, exist_ok=True)

now = datetime.now().strftime("%Y%m%d_%H%M%S")

logger.info("Initializing Pipeline...")
pipeline = SCETPipeline()


def benchmark_pipeline():
    
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
        
        # search
        start_time = time.time()
        df = pipeline.search.search_papers(query, limit=1000, alpha=0.5)
        search_time = time.time() - start_time
        if df.empty:
            logger.warning(f"No results for {query}")
            continue
            
        # cluster
        start_time = time.time()
        concepts, labels, n_clusters_used = pipeline.cluster.get_clustered_concepts(df, n_clusters=None)
        cluster_time = time.time() - start_time
        
        # era detection
        start_time = time.time()
        concept_eras = agg.identify_concept_eras(concepts, max_eras=5)
        era_time = time.time() - start_time
        
        # pivotal papers
        start_time = time.time()
        pivotals = agg.get_pivotal_papers(concepts, concept_eras, k=3)
        pivotal_time = time.time() - start_time
        
        # build result
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

    # output as csv
    os.makedirs(os.path.join(results_dir, 'pipeline'), exist_ok=True)
    out_file = os.path.join(results_dir, f"pipeline/{now}.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_file, index=False)
    logger.info(f"Benchmark complete. Results saved to {out_file}")



def benchmark_clustering():
    # Fetch a large pool of papers
    query = "machine learning"
    logger.info(f"Fetching pool of papers for query: '{query}'...")
    pool_df = pipeline.search.search_papers(query, limit=5000, alpha=0.5)
    logger.info(f"Found {len(pool_df)} papers in pool.")
    
    sample_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000]
    results = []
    
    for size in sample_sizes:
        if size > len(pool_df):
            logger.warning(f"Skipping size {size} (pool size: {len(pool_df)})")
            continue
            
        logger.info(f"Benchmarking clustering with {size} papers...")
        
        # get random sample from pool
        sample_df = pool_df.sample(n=size, random_state=42)
        
        # measure clustering time
        start_time = time.time()
        _, _, n_clusters_used = pipeline.cluster.get_clustered_concepts(sample_df)
        duration = time.time() - start_time
        
        results.append({
            "n_papers": size,
            "n_clusters_found": n_clusters_used,
            "time_sec": duration,
            "papers_per_sec": size / duration if duration > 0 else 0,
            "param_n_clusters": "Auto"
        })
        
    # output as csv
    os.makedirs(os.path.join(results_dir, 'clustering'), exist_ok=True)
    output_file = os.path.join(results_dir, f"clustering/{now}.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    logger.info(f"Benchmark complete. Results saved to {output_file}")

    # create plot
    logger.info("Generating Scalability Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['n_papers'], df_results['time_sec'], marker='o', linestyle='-', color='b')
    plt.title('Clustering Scalability (Time vs N Papers)')
    plt.xlabel('Number of Papers')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "clustering_scalability.png"))
    logger.info(f"Plot saved to {image_dir}/clustering_scalability.png")


def main():
    benchmark_pipeline()
    benchmark_clustering()

if __name__ == "__main__":
    main()
