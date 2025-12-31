# Scientific Concept Evolution Tracker

## Initial Proposal

This project proposes a methodology for tracking the semantic evolution of scientific ideas over time by leveraging temporal vector analysis. This system processes a large corpus of research papers, sourced from repositories like [ArXiv](https://arxiv.org/) and [PubMed Central](https://pmc.ncbi.nlm.nih.gov/), and generates contextual embeddings for documents across distinct time slices (e.g., decades). These time-stamped embeddings are stored in a vector database, enabling users to query a specific scientific concept, such as "dark matter" or "CRISPR". The system retrieves the most relevant papers from different eras, not merely to show a chronological list, but to reveal how the conceptual neighborhood and contextual meaning of the term have shifted. A temporal reranking mechanism can further prioritize papers that represent pivotal turning points in the concept's history. The key innovation is the creation of a "time machine" for scientific thought, providing historians of science and researchers with a powerful tool to visualize and analyze the dynamic life cycle of scientific concepts within the latent space.


## Get Started

### Building SCET from scratch

First, the respective systems must be deployed. SCET requires 500GB of storage and uses the following
| Tool                                    | Memory (GB) | CPUs   |
|-----------------------------------------|-------------|--------|
| [Postgres](https://www.postgresql.org/) | 2           | 2      |
| [Milvus](https://milvus.io/)            | 20          | 6      |
| [MinIO](https://www.min.io/)            | 4           | 1      |
| [etcd](https://etcd.io/)                | 2           | 1      |
| **Total**                               | **28**      | **10** |


This project uses [Docker](https://www.docker.com/) and [uv](https://docs.astral.sh/uv/). The entire system can be initialized with

```
docker compose up -d
```

> [!NOTE]
> In the report everything except for Postgres was deployed on an `m6gd.2xlarge` EC2. Postgres was left running on a homelab since it only requires any significant CPU/RAM on data ingestion.

Copy the example environment file and make edits if required

```bash
cp .env.example .env
```

Setup Postgres schemas

```bash
uv run ./src/init_db.py
```

Ingest the Kaggle data. You can either [download it directly from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) and supply the path to the JSON file with the `--data-path` argument, or you can run the [get_arxiv_data.ipynb](./notebooks/get_arxiv_data.ipynb) notebook (which will download for you). Once this is done, we put it into Postgres

```bash
# this will likely take some time to complete
uv run ./src/ingest_metadata.py
```

Then we generate the embeddings and put them into Milvus

```bash
# this will likely take some time to complete
uv run ./src/embed_milvus.py
```

> [!NOTE]
> We had access to a SLURM cluster, so we could speed this step up via `sbatch ./src/embed_milvus_slurm.sh`

Finally we create the indexes

```bash
uv run ./src/build_indices.py
```

The system is now ready for use.



### Streamlit UI

```bash
uv run streamlit run ./src/app.py
```