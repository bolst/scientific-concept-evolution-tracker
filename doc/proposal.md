# Proposal

## Introduction

Scientific terminology such as "CRISPR" or "dark matter" have contextual
associations and meanings with a potential to change over time. Vector
databases typically index semantics that are implicitly static and fail
to account for this natural semantic drift. The Scientific Concept
Evolution Tracker (SCET) will address this limitation by architecting a
retrieval system designed for time-dependent analysis.

Our research question is defined as follows: **How can a vector database
system be designed and implemented to reliably quantify the semantic
evolution of various scientific concepts over time, and can that be
extended to identify specific publications that are "pivotal" to a
scientific concept?**

## Relevant Background

Vector databases are built to manage high-dimensional vector embeddings.
This enables efficient Approximate Nearest Neighbor (ANN) search for
similarity retrieval, with an additional layer of complexity with
semantic drift (the changing associations of words over time).
Literature has identified various approaches to address this, such as
modelling latent trajectories with diffusion [1] or by
embedding temporal aspects in the feature space [2].

For implementation, a core ability (of the vector database) must be to
query specific temporal slices efficiently, given a scientific term.
Leveraging a form of indexing on the timestamp metadata is a clear
strategy to optimize time-based range-filtered queries, where the system
must retrieve relevant papers within a specific publication period
(e.g., 2005-2010).

## Objectives

The specific research objectives are as follows.

1.  *Temporal Data Processing:* Establish a reproducible pipeline to
    segment a large, time-stamped corpus (from sources such as ArXiv or
    PubMed Central) into chronologically distinct time slices and
    generate contextual document embeddings.

2.  *Quantification of Semantic Drift:* Select a set of high-interest
    scientific concepts (e.g., LLMs, gene editing) and investigate the
    best way to measure their semantic evolution, including
    visualizations of the trajectory of concept clusters in the aligned
    latent space.

## Methodology

### Data Preparation and Temporal Slicing

A corpus of time-stamped scientific paper abstracts (e.g., from ArXiv)
will be obtained. The corpus will be sequentially partitioned over a
fixed number of years (e.g., 10 years) to capture measurable shifts in
conceptual meaning. If there is a defined "era" for a concept (such as
the growth of LLMs in the 2020s), then that will be used. For generating
document embeddings, an efficient, high-quality pre-trained transformer
model (e.g., a lightweight Sentence-Transformer model) will be utilized
for its balance of speed and semantic quality.

### System Implementation

A vector database will be utilized for the implementation, primarily for
the native support of advanced metadata indexing. The time-aligned
document vectors, along with essential metadata (title, abstract text,
publication timestamp), will be added to the database. The timestamp
field will be indexed.

### Drift Analysis

The final analysis will quantify the semantic drift. We will calculate
some distance metric (e.g., cosine distance) between the concept
centroid vectors of adjacent time slices in the aligned space. Also, we
will visually map the concept centroids across time, providing a clear
visual representation of the quantified conceptual trajectory. We will
also investigate and analyze how a specific publication can be deemed as
"pivotal" (e.g., [3]).

## Expected Outcomes

The final deliverable will be a comprehensive report providing:

- *Quantitative Drift Analysis:* A measure of the semantic drift for the
  target concepts. These measures can be used to identify papers that
  are "pivotal" on a concepts history. Both of these will have an
  analysis performed and documented.

- *System Validation:* A working, benchmarked vector database pipeline
  and analysis scripts that successfully demonstrate the system's
  ability to retrieve and analyze time-aligned concepts, effectively
  functioning as a "time machine" for scientific thought.

# References

[1] R. Bamler and S. Mandt, “Dynamic word embeddings,” 2017. [Online]. Available: https://arxiv.org/abs/1702.08359

[2] R. C. Barranco, R. F. D. Santos, and M. S. Hossain, “Tracking the evolution
of words with time-reflective text representations,” 2019. [Online]. Available:
https://arxiv.org/abs/1807.04441

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
L. Kaiser, and I. Polosukhin, “Attention is all you need,” 2023. [Online]. Available:
https://arxiv.org/abs/1706.03762