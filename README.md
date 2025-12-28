# Scientific Concept Evolution Tracker

This project proposes a novel methodology for tracking the semantic evolution of scientific ideas over time by leveraging temporal vector analysis. This system processes a large corpus of research papers, sourced from repositories like [ArXiv](https://arxiv.org/) and [PubMed Central](https://pmc.ncbi.nlm.nih.gov/), and generates contextual embeddings for documents across distinct time slices (e.g., decades). These time-stamped embeddings are stored in a vector database, enabling users to query a specific scientific concept, such as "dark matter" or "CRISPR". The system retrieves the most relevant papers from different eras, not merely to show a chronological list, but to reveal how the conceptual neighborhood and contextual meaning of the term have shifted. A temporal reranking mechanism can further prioritize papers that represent pivotal turning points in the concept's history. The key innovation is the creation of a "time machine" for scientific thought, providing historians of science and researchers with a powerful tool to visualize and analyze the dynamic life cycle of scientific concepts within the latent space.

## Get Started

### Streamlit UI

```bash
uv run streamlit run ./src/app.py
```