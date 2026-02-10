import os

from carnot.data.dataset import Dataset
from carnot.operators.code import CodeOperator

TEST_MODEL_ID = "openai/gpt-5-mini"
LLM_CONFIG = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}

def test_code_operator_no_inputs():
    task = "what is 2 + 2?"
    code_operator = CodeOperator(task, "output-dataset-id", TEST_MODEL_ID, LLM_CONFIG)
    input_datasets = {}
    output_datasets = code_operator(input_datasets)

    # check that there is one output dataset
    assert len(output_datasets) == 1
    assert "output-dataset-id" in output_datasets

    # check that the code state contains the correct result
    output_state = output_datasets["output-dataset-id"].code_state
    assert list(output_state.values())[0] == 4


def test_code_operator_one_dataset(movie_reviews_data):
    movies_df, _ = movie_reviews_data

    # create movies dataset
    movies_dataset = Dataset(
        name="Movies Dataset",
        annotation="A dataset containing information about various movies.",
        items=movies_df.to_dict(orient="records"),
    )

    # create input to code operator
    task = "Return the list of movie titles which made more than 50 million dollars at the box office."
    input_datasets = {"Movies Dataset": movies_dataset}

    # generate output
    code_operator = CodeOperator(task, "output-dataset-id", TEST_MODEL_ID, LLM_CONFIG)
    output_datasets = code_operator(input_datasets)

    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_state = output_datasets["output-dataset-id"].code_state
    output_movies = str(list(output_state.values())[0]).lower()
    assert "inception" in output_movies
    assert "mean girls" in output_movies
    assert "volver" not in output_movies

def test_code_operator_two_datasets(movie_reviews_data):
    # load movie reviews data
    movies_df, reviews_df = movie_reviews_data

    # filter reviews_df to have 5 reviews per movie
    filtered_reviews_df = reviews_df.groupby("id").head(5).reset_index(drop=True)

    # create movies and reviews datasets with 5 reviews per movie
    movies_dataset = Dataset(
        name="Movies Dataset",
        annotation="A dataset containing information about various movies.",
        items=movies_df.to_dict(orient="records"),
    )
    reviews_dataset = Dataset(
        name="Reviews Dataset",
        annotation="A dataset containing reviews for various movies.",
        items=filtered_reviews_df.to_dict(orient="records"),
    )

    # create input to code operator
    task = "Return the list of movies that have at least one review with negative sentiment."
    input_datasets = {"Movies Dataset": movies_dataset, "Reviews Dataset": reviews_dataset}

    # generate output
    code_operator = CodeOperator(task, "output-dataset-id", TEST_MODEL_ID, LLM_CONFIG)
    output_datasets = code_operator(input_datasets)

    assert len(output_datasets) == 3
    assert "output-dataset-id" in output_datasets
    output_state = output_datasets["output-dataset-id"].code_state
    output_movies = str(list(output_state.values())[0]).lower()
    assert "inception" in output_movies
    assert "mean girls" not in output_movies and "mean_girls" not in output_movies
    assert "volver" not in output_movies
