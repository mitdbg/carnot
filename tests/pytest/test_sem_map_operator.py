from carnot.data.dataset import Dataset
from carnot.operators.sem_map import SemMapOperator

TEST_MODEL_ID = "openai/gpt-5-mini"

def test_sem_map_operator_basic():
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

    # create input to semantic map operator
    task = "Extract the group classification of the animal ([mammal, bird, reptile, fish, amphibian])"
    output_fields = [{"name": "animal_group", "type": str, "description": "The group classification of the animal."}]
    input_datasets = {animal_dataset.name: animal_dataset}

    # execute the operator
    sem_map_operator = SemMapOperator(task, output_fields, TEST_MODEL_ID, max_workers=4)
    output_datasets = sem_map_operator("Animal Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "SemMapOperatorOutput" in output_datasets
    output_dataset = output_datasets["SemMapOperatorOutput"]
    assert len(output_dataset.items) == 5
    assert {"animal": "giraffe", "animal_group": "mammal"} in output_dataset.items
    assert {"animal": "anaconda", "animal_group": "reptile"} in output_dataset.items
    assert {"animal": "salmon", "animal_group": "fish"} in output_dataset.items
    assert {"animal": "elephant", "animal_group": "mammal"} in output_dataset.items
    assert {"animal": "tucan", "animal_group": "bird"} in output_dataset.items


def test_sem_map_operator_movie_reviews(movie_reviews_data):
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
    task = "The sentiment of the review (positive or negative)"
    output_fields = [{"name": "sentiment", "type": str, "description": "The sentiment of the review, can be either 'positive' or 'negative'."}]
    input_datasets = {"Reviews Dataset": reviews_dataset}

    # generate output
    sem_map_operator = SemMapOperator(task, output_fields, TEST_MODEL_ID, max_workers=4)
    output_datasets = sem_map_operator("Reviews Dataset", input_datasets)

    assert len(output_datasets) == 2
    assert "SemMapOperatorOutput" in output_datasets
    output_dataset = output_datasets["SemMapOperatorOutput"]
    assert len(output_dataset.items) == 15
    total_correct, total = 0, 0
    for review in filtered_reviews_df.to_dict(orient="records"):
        gt_sentiment = review["scoreSentiment"].lower()
        review_id = review["id"]
        for output_review in output_dataset.items:
            if output_review["id"] == review_id:
                total_correct += output_review["sentiment"].lower() == gt_sentiment
                total += 1
    accuracy = total_correct / total
    assert accuracy >= 0.8
