import os
import time
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from pipeline import SCETPipeline
import utilities.aggregate as agg
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(root_path, "benchmarks")
os.makedirs(results_dir, exist_ok=True)
image_dir = os.path.join(root_path, "doc", "report", "images")
os.makedirs(image_dir, exist_ok=True)
doc_results_dir = os.path.join(root_path, "doc", "report", "benchmarks")
os.makedirs(doc_results_dir, exist_ok=True)

now = datetime.now().strftime("%Y%m%d_%H%M%S")

logger.info("Initializing Pipeline...")
pipeline = SCETPipeline()


def _jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0



def benchmark_pipeline():
    
    queries = [
        "Transformer",
        "Corona",
        "Entropy",
        "Token",
        "Deep Learning",
        "CRISPR",
        "Climate Change",
        "Diffusion",
        "Vector"
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
    # output summary
    out_file = os.path.join(results_dir, f"pipeline/{now}_summary.csv")
    total_time = df_results.search_time_sec + df_results.cluster_time_sec + df_results.era_detection_time_sec + df_results.pivotal_papers_time_sec
    df_avg = pd.DataFrame(columns=['stage', 'avg_time_sec'])
    df_avg.loc[0] = ['Hybrid Search', np.average(df_results.search_time_sec)]
    df_avg.loc[1] = ['Concept Clustering', np.average(df_results.cluster_time_sec)]
    df_avg.loc[2] = ['Era Detection', np.average(df_results.era_detection_time_sec)]
    df_avg.loc[3] = ['Pivotal Paper ID', np.average(df_results.pivotal_papers_time_sec)]
    df_avg['percent_time_sec'] = df_avg.avg_time_sec / df_avg.avg_time_sec.sum() * 100
    df_avg.loc[4] = ['Total', df_avg.avg_time_sec.sum(), df_avg.percent_time_sec.sum()]
    df_avg.to_csv(out_file, index=False)
    
    logger.info(f"Pipeline benchmark complete.")



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
    logger.info(f"Clustering benchmark complete.")

    # create plot
    logger.info("Generating Cluster Scalability Plot...")
    plt.figure()
    plt.plot(df_results['n_papers'], df_results['time_sec'], marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Papers')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "clustering_scalability.png"))
    logger.info(f"Plot saved to {image_dir}/clustering_scalability.png")


def benchmark_ablation():
    queries = [
        "Transformer",
        "Corona",
        "Entropy",
        "Token",
        "Attention",
        "Neural Networks",
        "Diffusion"
    ]
    
    summary_results = []
    detailed_results = []
    
    logger.info("Starting Ablation study (Hybrid vs Dense vs Sparse)...")
    
    for query in queries:
        logger.info(f"Processing query: {query}")
        
        # Dense
        df_dense = pipeline.search.search_papers(query, limit=50, alpha=1.0)
        ids_dense = df_dense['arxiv_id'].tolist() if not df_dense.empty else []
        # Sparse
        df_sparse = pipeline.search.search_papers(query, limit=50, alpha=0.0)
        ids_sparse = df_sparse['arxiv_id'].tolist() if not df_sparse.empty else []
        # Hybrid
        df_hybrid = pipeline.search.search_papers(query, limit=50, alpha=0.5)
        ids_hybrid = df_hybrid['arxiv_id'].tolist() if not df_hybrid.empty else []

        # calculate overlaps
        jaccard_dense_hybrid = _jaccard_similarity(ids_dense, ids_hybrid)
        jaccard_sparse_hybrid = _jaccard_similarity(ids_sparse, ids_hybrid)
        jaccard_dense_sparse = _jaccard_similarity(ids_dense, ids_sparse)
        
        summary_results.append({
            "query": query,
            "jaccard_dense_hybrid": jaccard_dense_hybrid,
            "jaccard_sparse_hybrid": jaccard_sparse_hybrid,
            "jaccard_dense_sparse": jaccard_dense_sparse,
            "n_results_dense": len(ids_dense),
            "n_results_sparse": len(ids_sparse),
            "n_results_hybrid": len(ids_hybrid),
            "param_limit": 50,
            "param_alpha_dense": 1.0,
            "param_alpha_sparse": 0.0,
            "param_alpha_hybrid": 0.5
        })
        
        # store top 10 for each type
        for method, df in [("Dense", df_dense), ("Sparse", df_sparse), ("Hybrid", df_hybrid)]:
            for i, row in df.head(10).iterrows():
                detailed_results.append({
                    "query": query,
                    "method": method,
                    "rank": i+1,
                    "arxiv_id": row['arxiv_id'],
                    "title": row.get('title', ''),
                    "year": row.get('publication_year', ''),
                    "score": row['score']
                })

    # save results
    os.makedirs(os.path.join(results_dir, 'ablation'), exist_ok=True)
    pd.DataFrame(summary_results).to_csv(os.path.join(results_dir, f"ablation/{now}_summary.csv"), index=False)
    pd.DataFrame(detailed_results).to_csv(os.path.join(results_dir, f"ablation/{now}_detail.csv"), index=False)
    logger.info(f"Ablation benchmark complete.")

    logger.info("Generating Ablation Plot...")
    df_summary = pd.DataFrame(summary_results)
    queries = df_summary['query']
    x = np.arange(len(queries))
    width = 0.25
    plt.figure()
    plt.bar(x - width, df_summary['jaccard_dense_hybrid'], width, label="$\\text{Dense} \\cap \\text{Hybrid}$")
    plt.bar(x, df_summary['jaccard_sparse_hybrid'], width, label="$\\text{Sparse} \\cap \\text{Hybrid}$")
    plt.bar(x + width, df_summary['jaccard_dense_sparse'], width, label="$\\text{Dense} \\cap \\text{Sparse}$")
    
    plt.xlabel('Query')
    plt.ylabel('Jaccard Similarity')
    plt.xticks(x, queries, rotation=45)
    plt.legend()
    plt.tight_layout()
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, "ablation_overlap.png"))
    logger.info(f"Plot saved to {image_dir}/ablation_overlap.png")


def copy_to_report():
    # define files to copy
    files = [
        (f"pipeline/{now}_summary.csv", "pipeline_summary.csv"),
        (f"clustering/{now}.csv", "clustering.csv"),
        (f"ablation/{now}_summary.csv", "ablation.csv"),
    ]
    # copy files
    for src, dest in files:
        shutil.copy(
            os.path.join(results_dir, src), 
            os.path.join(doc_results_dir, dest)
            )

def main():
    benchmark_pipeline()
    benchmark_clustering()
    benchmark_ablation()

    copy_to_report()

if __name__ == "__main__":
    main()
