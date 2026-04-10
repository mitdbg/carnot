import pytest

from carnot.data.dataset import Dataset
from carnot.operators.sem_agg import SemAggOperator


@pytest.mark.llm
def test_sem_agg_operator_basic(test_model_id, llm_config):
    # construct dataset of various animals
    animal_data = [
        {"animal": animal}
        for animal in ["giraffe", "anaconda", "salmon", "elephant", "tucan"]
    ]
    animal_dataset = Dataset(
        name="Animal Dataset",
        annotation="A dataset containing different animals.",
        items=animal_data,
    )

    # create input to semantic agg. operator
    task = "The largest animal by weight"
    output_fields = [{"name": "largest_animal", "type": str, "description": "The largest animal by weight."}]
    input_datasets = {animal_dataset.name: animal_dataset}

    # execute the operator
    sem_agg_operator = SemAggOperator(task, output_fields, "output-dataset-id", test_model_id, llm_config, max_workers=4)
    output_datasets, _stats = sem_agg_operator("Animal Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 1
    assert {"largest_animal": "elephant"} in output_dataset.items


@pytest.mark.llm
def test_sem_agg_operator_movie_reviews(test_model_id, llm_config, movie_reviews_data):
    # load movie reviews data
    _, reviews_df = movie_reviews_data

    # filter reviews_df to have 5 reviews per movie
    filtered_reviews_df = reviews_df.groupby("id").head(5).reset_index(drop=True)

    # create reviews datasets with 5 reviews per movie
    reviews_dataset = Dataset(
        name="Reviews Dataset",
        annotation="A dataset containing reviews for various movies.",
        items=filtered_reviews_df.to_dict(orient="records"),
    )

    # create input to code operator
    task = "The movie containing the most negative reviews"
    output_fields = [{"name": "worst_movie", "type": str, "description": "The movie with the most negative reviews."}]
    input_datasets = {"Reviews Dataset": reviews_dataset}

    # generate output
    sem_agg_operator = SemAggOperator(task, output_fields, "output-dataset-id", test_model_id, llm_config, max_workers=4)
    output_datasets, _stats = sem_agg_operator("Reviews Dataset", input_datasets)

    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 1
    assert {"worst_movie": "inception"} in output_dataset.items
