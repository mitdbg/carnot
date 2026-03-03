import os
from pathlib import Path

import pandas as pd
import pytest

from carnot.data.dataset import Dataset
from carnot.data.item import DataItem
from carnot.index import FlatCarnotIndex, HierarchicalCarnotIndex


@pytest.fixture
def movie_reviews_data():
    base_dir = (
        "tests/pytest/data/movie-reviews/4"
        if os.path.exists("tests/pytest/data/movie-reviews/4")
        else "tests/pytest/data/movie-reviews/"
    )
    movies_df = pd.read_csv(f"{base_dir}/rotten_tomatoes_movies.csv")
    reviews_df = pd.read_csv(f"{base_dir}/rotten_tomatoes_movie_reviews.csv")

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


@pytest.fixture
def enron_emails_data():
    emails = []
    for email in os.listdir("tests/pytest/data/emails"):
        with open(f"tests/pytest/data/emails/{email}") as f:
            contents = f.read()
            emails.append({"contents": contents})
    return emails


@pytest.fixture
def enron_data_items():
    """DataItems with paths to Enron email files (for Flat/Hierarchical indices)."""
    enron_dir = Path(__file__).resolve().parent.parent / "data" / "emails"
    if not enron_dir.exists():
        pytest.skip(f"Enron data dir not found: {enron_dir}")
    return [DataItem(path=str(p.absolute())) for p in sorted(enron_dir.glob("*.txt"))]


@pytest.fixture
def enron_data_items_small(enron_data_items):
    """Subset of 30 Enron files for faster index builds in E2E tests."""
    return enron_data_items[:30]


@pytest.fixture
def small_enron_dataset_with_index(llm_config, enron_data_items_small):
    """Dataset with pre-built Flat index for E2E index-aware planning tests."""
    flat_index = FlatCarnotIndex(
        name="enron-flat",
        items=enron_data_items_small,
        api_key=llm_config.get("OPENAI_API_KEY"),
    )
    return Dataset(
        name="Enron Emails",
        annotation="Enron email corpus for semantic search.",
        items=enron_data_items_small,
        indices={"flat": flat_index},
    )

@pytest.fixture
def enron_dataset_with_flat_index(llm_config, enron_data_items):
    """Dataset with pre-built Flat index for E2E index-aware planning tests."""
    from carnot.data.dataset import Dataset
    from carnot.index import FlatCarnotIndex

    flat_index = FlatCarnotIndex(
        name="enron-flat-big",
        items=enron_data_items,
        api_key=llm_config.get("OPENAI_API_KEY"),
    )

    return Dataset(
        name="Enron Emails Big",
        annotation="Full Enron email corpus for semantic search.",
        items=enron_data_items,
        indices={"flat": flat_index},
    )

@pytest.fixture
def enron_dataset_with_hierarchical_index(llm_config, enron_data_items):
    """Dataset with pre-built Flat index for E2E index-aware planning tests."""
    hierarchical_index = HierarchicalCarnotIndex(
        name="enron-hierarchical",
        items=enron_data_items,
        api_key=llm_config.get("OPENAI_API_KEY"),
    )

    return Dataset(
        name="Enron Emails Hierarchical",
        annotation="Full Enron email corpus for semantic search.",
        items=enron_data_items,
        indices={"hierarchical": hierarchical_index},
    )
    