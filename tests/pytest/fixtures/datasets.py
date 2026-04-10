"""
Fixtures for logical planning tests.
"""
import pytest

from carnot.data.dataset import Dataset
from carnot.data.item import DataItem


@pytest.fixture
def movie_reviews_datasets(movie_reviews_data):
    """
    Load movie review datasets from CSV files.
    Returns a tuple of (movies_dataset, reviews_dataset).
    """
    filtered_movies_df, filtered_reviews_df = movie_reviews_data

    # Limit to 10 reviews per movie
    filtered_reviews_df = filtered_reviews_df.groupby("id").head(10).reset_index(drop=True)

    # Create datasets
    movies_dataset = Dataset(
        name="Movies",
        annotation="A dataset containing movie metadata including title, ratings, genre, director, etc.",
        items=[DataItem.from_dict(row.to_dict()) for _, row in filtered_movies_df.iterrows()],
    )

    reviews_dataset = Dataset(
        name="Reviews",
        annotation="A dataset containing movie reviews with critic names, review text, and sentiment scores.",
        items=[DataItem.from_dict(row.to_dict()) for _, row in filtered_reviews_df.iterrows()],
    )

    return movies_dataset, reviews_dataset


@pytest.fixture
def simple_movie_dataset():
    """
    Create a simple in-memory movie dataset for basic tests.
    """
    movies_data = [
        {
            "id": "inception",
            "title": "Inception",
            "genre": "Sci-Fi",
            "director": "Christopher Nolan",
            "rating": 8.8,
            "year": 2010,
        },
        {
            "id": "interstellar",
            "title": "Interstellar",
            "genre": "Sci-Fi",
            "director": "Christopher Nolan",
            "rating": 8.6,
            "year": 2014,
        },
        {
            "id": "the_prestige",
            "title": "The Prestige",
            "genre": "Mystery",
            "director": "Christopher Nolan",
            "rating": 8.5,
            "year": 2006,
        },
        {
            "id": "pulp_fiction",
            "title": "Pulp Fiction",
            "genre": "Crime",
            "director": "Quentin Tarantino",
            "rating": 8.9,
            "year": 1994,
        },
    ]

    return Dataset(
        name="Movies",
        annotation="A simple dataset of popular movies with metadata",
        items=[DataItem.from_dict(movie) for movie in movies_data],
    )
