import os

import pandas as pd
import pytest


@pytest.fixture
def movie_reviews_data():
    movies_df = pd.read_csv("tests/pytest/data/movie-reviews/4/rotten_tomatoes_movies.csv")
    reviews_df = pd.read_csv("tests/pytest/data/movie-reviews/4/rotten_tomatoes_movie_reviews.csv")

    movie_ids = ["inception", "volver", "mean_girls"]
    filtered_movies_df = movies_df[movies_df["id"].isin(movie_ids)]
    filtered_reviews_df = reviews_df[reviews_df["id"].isin(movie_ids)]

    return filtered_movies_df, filtered_reviews_df


@pytest.fixture
def research_papers_data():
    papers = []
    for paper in os.listdir("tests/pytest/data/papers"):
        with open(f"tests/pytest/data/papers/{paper}") as f:
            contents = f.read()
            papers.append({"contents": contents})

    return papers
