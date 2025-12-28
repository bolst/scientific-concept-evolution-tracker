import matplotlib.pyplot as plt
from pipeline import SCETPipeline
import streamlit as st

# =================================================================

st.set_page_config(page_title="SCET: Scientific Concept Evolution Tracker", layout="wide")

@st.cache_resource
def get_pipeline():
    return SCETPipeline()

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to initialize pipeline: {e}")
    st.stop()

st.title("Scientific Concept Evolution Tracker")
st.markdown(
"""
**Track how scientific concepts evolve over time.**  
Enter a query to find relevant papers, cluster them into sub-concepts, and identify pivotal papers.
"""
)

# set up sidebar
with st.sidebar:
    st.header("Configuration")
    # alpha
    alpha = st.slider("Search Weight $\\alpha$", 0.0, 1.0, 0.5, help="0.0 = keyword only (sparse), 1.0 = semantic only (dense)")
    # date ranges
    start_year, end_year = st.slider("Time Range", 1980, 2025, (1980, 2025))
    # clusters
    n_clusters_option = st.selectbox("Number of Clusters", ["Auto", "Manual"])
    n_clusters = st.slider("Clusters", 2, 10, 3) if n_clusters_option == "Manual" else None
    # max eras
    max_eras = st.slider("Max Eras per Concept", 1, 10, 5)
    # pivotal papers per era
    pivotal_k = st.slider("Pivotal Papers per Era", 1, 10, 3)

# set up query
query = st.text_input("Search Query", "Entropy")

# main action button
if st.button("Analyze Concept", type="primary"):
    with st.spinner("Running analysis... (Searching -> Clustering -> Era Detection)"):
        try:
            # run search
            result = pipeline.process(
                query=query,
                alpha=alpha,
                start_year=start_year,
                end_year=end_year,
                n_clusters=n_clusters,
                max_eras=max_eras,
                pivotal_papers_per_era=pivotal_k
            )
            st.success(f"Analysis complete! Found {len(result.search_results)} papers.")
            
            # create concept evolution plot
            st.subheader("Concept Evolution Over Time")
            df = result.concept_clusters
            if not df.empty:
                counts = df.groupby(['year', 'cluster_label']).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(12, 6))
                counts.plot(kind='bar', stacked=True, ax=ax, width=0.9)
                ax.set_xlabel("Year")
                ax.set_ylabel("Number of Papers")
                ax.legend(title="Concepts")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data to plot.")
            
            # eras and pivotal papers
            st.subheader("Concept Eras & Pivotal Papers")
            if not result.pivotal_papers:
                st.info("No concepts identified.")
            # create tabs for each concept
            concept_labels = list(result.pivotal_papers.keys())
            if concept_labels:
                tabs = st.tabs(concept_labels)
                for i, label in enumerate(concept_labels):
                    with tabs[i]:
                        era_pivotal_list = result.pivotal_papers[label]
                        # show summary for each era in concept
                        for era_data in era_pivotal_list:
                            with st.container():
                                st.markdown(f"### Era: {era_data.period}")
                                st.caption(f"Average Activity: {era_data.era.avg_count:.1f} papers/year")
                                for p in era_data.papers:
                                    st.markdown(f"""
                                    - **{p.year}** | [arXiv:{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})  
                                      **{p.title}**  
                                      *(Score: {p.score:.4f})*
                                    """)
                                st.divider()

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.exception(e)
