from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict
import pandas as pd

from dataclasses import dataclass

@dataclass
class EraStat:
    start: int
    end: int
    avg_count: float

@dataclass
class EraPaper:
    title: str
    year: int
    score: float
    arxiv_id: str

@dataclass
class EraPivotalPapers:
    era: EraStat
    period: str
    papers: list[EraPaper]


def identify_concept_eras(df: pd.DataFrame, max_eras: int = 5) -> dict[str, list[EraStat]]:
    eras = {}
    labels = df['cluster_label'].unique()
    for label in labels:
        # get counts by year
        counts = df[df['cluster_label'] == label].groupby('year').size()
        if len(counts) < 2:
            continue
            
        # fill missing years with 0
        min_year, max_year = counts.index.min(), counts.index.max()
        counts = counts.reindex(range(min_year, max_year + 1), fill_value=0)
        
        # build and fit decision tree
        years = counts.index.values.reshape(-1, 1)
        yearly_count = counts.values
        tree = DecisionTreeRegressor(max_leaf_nodes=max_eras)
        tree.fit(years, yearly_count)
        
        # extract thresholds (change points) from the tree
        # we only care about non-leaf nodes (i.e., where feature != -2)
        thresholds = tree.tree_.threshold[tree.tree_.feature != -2]
        change_points = sorted([int(t) for t in thresholds])
        
        # edges of each era (e.g., if eras are [2000-2007, 2008-2014] then 2007, 2014)
        # last year won't be included, so include manually
        era_edges = change_points + [max_year]
        # build eras
        eras[label] = []
        start_year = min_year
        for end_year in era_edges:
            avg_count = counts[(counts.index >= start_year) & (counts.index <= end_year)].mean()
            stat = EraStat(start = start_year, end = end_year, avg_count = avg_count)
            eras[label].append(stat)
            start_year = end_year + 1
        
    return eras


def get_pivotal_papers(
    cluster_df: pd.DataFrame, 
    eras: dict[str, list[EraStat]], 
    k: int = 3
    ) -> dict[str, list[EraPivotalPapers]]:
    
    pivotal_papers = defaultdict(list)
    for label, era_list in eras.items():
        for era in era_list:
            # filter for both the concept and era
            mask = (
                (cluster_df['cluster_label'] == label) & 
                (cluster_df['year'] >= era.start) & 
                (cluster_df['year'] <= era.end)
            )
            era_papers = cluster_df[mask].sort_values('score', ascending=False).head(k)
            
            # build data structure of pivotal papers
            papers_info = [EraPaper(
                title = row.title,
                year = row.year,
                score = row.score,
                arxiv_id = row.arxiv_id    
            ) for _, row in era_papers.iterrows()]
            pivotal_papers[label].append(EraPivotalPapers(
                era = era,
                period = f"{era.start}-{era.end}",
                papers = papers_info
            ))
            
    return pivotal_papers