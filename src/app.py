import streamlit as st
import pandas as pd
import plotly.express as px

from scet.search import search, analyze_evolution

def main():
    st.set_page_config(page_title="SCET: Time Machine", layout="wide")

    st.title("🕰️ Scientific Concept Evolution Tracker")
    st.markdown("""
    **Visualize the life cycle of scientific ideas.**  
    See how the meaning, context, and disciplinary home of a concept shifts over time.
    """)

    # Sidebar Config
    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Mode", ["Evolution Analysis", "Paper Search"])

    if mode == "Evolution Analysis":
        st.header("📈 Concept Evolution Analysis")
        query = st.text_input("Enter a Concept", placeholder="e.g., 'attention', 'entropy', 'neural networks'")
        
        if st.button("Analyze Evolution"):
            if not query:
                st.warning("Please enter a concept.")
            else:
                with st.spinner(f"Tracking '{query}' through time..."):
                    # Run analysis
                    data = analyze_evolution(query, start_year=2000, end_year=2024)
                    
                    if not data:
                        st.error("No data found for this concept in the sampled timeline.")
                    else:
                        # Process data for visualization
                        df_evolution = pd.DataFrame(data)
                        
                        # 1. Relevance Over Time
                        st.subheader("1. Semantic Relevance Over Time")
                        st.caption("How closely did papers in each year match your modern query?")
                        fig_rel = px.line(df_evolution, x="year", y="avg_score", markers=True, 
                                        labels={"avg_score": "Semantic Similarity (Avg)", "year": "Year"})
                        
                        st.plotly_chart(fig_rel, use_container_width=True)
                        
                        # 2. Category Evolution (The "Drift")
                        st.subheader("2. Disciplinary Drift (Category Evolution)")
                        st.caption("Which scientific fields were discussing this concept?")
                        
                        # Flatten categories for counting
                        cat_data = []
                        for entry in data:
                            year = entry['year']
                            for cat in entry['categories']:
                                # Group sub-categories (e.g., cs.LG -> cs) for cleaner chart? 
                                # Or keep full for detail. Let's keep full but maybe limit to top 5 overall.
                                cat_data.append({"year": year, "category": cat})
                        
                        df_cat = pd.DataFrame(cat_data)
                        # Count category occurrences per year
                        df_cat_counts = df_cat.groupby(['year', 'category']).size().reset_index(name='count')
                        
                        # Calculate percentage per year
                        df_cat_counts['percentage'] = df_cat_counts.groupby('year')['count'].transform(lambda x: x / x.sum())
                        
                        fig_area = px.area(df_cat_counts, x="year", y="percentage", color="category",
                                        labels={"percentage": "Share of Discussion", "year": "Year"},
                                        title="Evolution of Disciplinary Context")
                        st.plotly_chart(fig_area, use_container_width=True)

    elif mode == "Paper Search":
        st.header("🔎 Deep Dive Search")
        limit = st.sidebar.slider("Results Limit", 5, 50, 10)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Search Query", placeholder="e.g., 'attention mechanism'")
        with col2:
            year = st.number_input("Filter by Year", min_value=1990, max_value=2025, value=None, step=1)

        if st.button("Search"):
            if not query:
                st.warning("Please enter a query.")
            else:
                with st.spinner(f"Searching..."):
                    search_year = int(year) if year else None
                    results = search(query, year=search_year, limit=limit, print_results=False)
                    
                    if not results:
                        st.info("No results found.")
                    else:
                        st.success(f"Found {len(results)} papers.")
                        for row in results:
                            with st.expander(f"[{row['year']}] {row['title']} (Score: {row['score']:.2f})"):
                                st.markdown(f"**Category:** `{row['category']}`")
                                st.markdown(f"**ArXiv ID:** `{row['arxiv_id']}`")
                                st.markdown(f"**Abstract:** {row['abstract']}")

    st.markdown("---")
    st.caption("Powered by Milvus, SPECTER2, and ArXiv.")

if __name__ == "__main__":
    main()