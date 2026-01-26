from carnot.data.dataset import Dataset
from carnot.operators.sem_flat_map import SemFlatMapOperator

TEST_MODEL_ID = "openai/gpt-5-mini"

def test_sem_flat_map_operator_basic():
    # construct dataset of various fruits
    fruit_data = [
        {"text": "Apple is a sweet red fruit. Banana is a long yellow fruit. Cherry is a small red fruit. Orange is a round orange fruit. Grape is a small purple fruit."},
    ]
    fruit_dataset = Dataset(
        name="Fruit Dataset",
        annotation="A dataset containing text about various fruits.",
        items=fruit_data,
    )

    # create input to semantic map operator
    task = "Extract the each fruit and its color."
    output_fields = [
        {"name": "fruit", "type": str, "description": "The name of the fruit."},
        {"name": "color", "type": str, "description": "The color of the fruit."},
    ]
    input_datasets = {fruit_dataset.name: fruit_dataset}

    # execute the operator
    sem_flat_map_operator = SemFlatMapOperator(task, output_fields, TEST_MODEL_ID, max_workers=4)
    output_datasets = sem_flat_map_operator("Fruit Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "SemFlatMapOperatorOutput" in output_datasets
    output_dataset = output_datasets["SemFlatMapOperatorOutput"]
    assert len(output_dataset.items) == 5
    for item in output_dataset.items:
        item["fruit"] = item["fruit"].lower()
        item["color"] = item["color"].lower()
    assert {"fruit": "apple", "color": "red"} in output_dataset.items
    assert {"fruit": "banana", "color": "yellow"} in output_dataset.items
    assert {"fruit": "cherry", "color": "red"} in output_dataset.items
    assert {"fruit": "orange", "color": "orange"} in output_dataset.items
    assert {"fruit": "grape", "color": "purple"} in output_dataset.items


def test_sem_flat_map_operator_movie_reviews(research_papers_data):
    # load movie reviews data
    papers = research_papers_data

    # create papers datasets
    papers_dataset = Dataset(
        name="Research Papers Dataset",
        annotation="A dataset containing the contents of various research papers.",
        items=papers,
    )

    # create input to code operator
    task = "Extract each author, their affiliation, and email address from the paper contents."
    output_fields = [
        {"name": "author", "type": str, "description": "The author's full name."},
        {"name": "affiliation", "type": str, "description": "The author's affiliation."},
        {"name": "email", "type": str, "description": "The author's email address."},
    ]
    input_datasets = {"Research Papers Dataset": papers_dataset}

    # generate output
    sem_flat_map_operator = SemFlatMapOperator(task, output_fields, TEST_MODEL_ID, max_workers=4)
    output_datasets = sem_flat_map_operator("Research Papers Dataset", input_datasets)

    assert len(output_datasets) == 2
    assert "SemFlatMapOperatorOutput" in output_datasets
    output_dataset = output_datasets["SemFlatMapOperatorOutput"]
    assert len(output_dataset.items) == 2 + 15 + 10
