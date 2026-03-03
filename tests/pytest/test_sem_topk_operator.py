import pytest

from carnot.data.dataset import Dataset
from carnot.operators.sem_topk import SemTopKOperator


@pytest.mark.llm
def test_sem_topk_operator_basic(test_embedding_model_id, llm_config):
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

    # create input to semantic top-k operator
    task = "Retrieve all animals which are mammals"
    input_datasets = {animal_dataset.name: animal_dataset}

    # execute the operator
    sem_topk_operator = SemTopKOperator(task, k=2, output_dataset_id="output-dataset-id", model_id=test_embedding_model_id, llm_config=llm_config, max_workers=4)
    output_datasets = sem_topk_operator("Animal Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 2
    assert {"animal": "giraffe"} in output_dataset.items
    assert {"animal": "elephant"} in output_dataset.items


@pytest.mark.llm
def test_sem_topk_operator_movie_reviews(test_embedding_model_id, llm_config, movie_reviews_data):
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
    task = "The review has positive sentiment"
    input_datasets = {"Reviews Dataset": reviews_dataset}

    # generate output
    sem_topk_operator = SemTopKOperator(task, k=5, output_dataset_id="output-dataset-id", model_id=test_embedding_model_id, llm_config=llm_config, max_workers=4)
    output_datasets = sem_topk_operator("Reviews Dataset", input_datasets)

    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 5
    total_correct, total = 0, 0
    for review in output_dataset.items:
        if review['scoreSentiment'].lower() == 'positive':
            total_correct += 1
        total += 1
    accuracy = total_correct / total
    assert accuracy >= 0.8

@pytest.mark.llm
def test_sem_topk_operator_with_index(test_embedding_model_id, llm_config, enron_data_items):
    """SemTopKOperator with Flat index finds Raptor-related emails."""
    emails_dataset = Dataset(
        name="Emails Dataset",
        annotation="Enron emails for semantic search.",
        items=enron_data_items,
    )
    task = "Find emails about the Raptor investment or LJM partnerships"
    input_datasets = {"Emails Dataset": emails_dataset}
    sem_topk_operator = SemTopKOperator(
        task=task,
        k=10,
        output_dataset_id="output-dataset-id",
        model_id=test_embedding_model_id,
        llm_config=llm_config,
        max_workers=4,
        index_name="flat",
    )
    output_datasets = sem_topk_operator("Emails Dataset", input_datasets)
    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 10
    result_paths = [getattr(i, "path", i.get("path", "") if isinstance(i, dict) else "") for i in output_dataset.items]
    raptor_related = ["kaminski-v", "delainey-d-sent", "whalley-g-merchant", "parks-j-deleted"]
    matches = sum(1 for p in result_paths for r in raptor_related if r in str(p))
    assert matches >= 2, f"Expected at least 2 Raptor-related files in top 10, got {matches}"
