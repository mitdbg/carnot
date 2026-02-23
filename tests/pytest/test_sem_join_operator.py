from carnot.data.dataset import Dataset
from carnot.operators.sem_join import SemJoinOperator


def test_sem_join_operator_basic(test_model_id, llm_config):
    # construct two datasets with various animals and sounds
    animal_data = [
        {"animal": animal}
        for animal in ["cow", "dog", "cat", "sheep", "pig"]
    ]
    animal_dataset = Dataset(
        name="Animal Dataset",
        annotation="A dataset containing different animals.",
        items=animal_data,
    )
    sound_data = [
        {"sound": sound}
        for sound in ["moo", "woof", "meow", "baaa", "oink"]
    ]
    sound_dataset = Dataset(
        name="Sound Dataset",
        annotation="A dataset containing different sounds.",
        items=sound_data,
    )

    # create input to semantic join operator
    task = "The animal makes the given sound"
    input_datasets = {animal_dataset.name: animal_dataset, sound_dataset.name: sound_dataset}

    # execute the operator
    sem_join_operator = SemJoinOperator(task, test_model_id, llm_config, "output-dataset-id", max_workers=4)
    output_datasets = sem_join_operator("Animal Dataset", "Sound Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 3
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 5
    assert {"animal": "cow", "sound": "moo"} in output_dataset.items
    assert {"animal": "dog", "sound": "woof"} in output_dataset.items
    assert {"animal": "cat", "sound": "meow"} in output_dataset.items
    assert {"animal": "sheep", "sound": "baaa"} in output_dataset.items
    assert {"animal": "pig", "sound": "oink"} in output_dataset.items

def test_sem_join_operator_papers(test_model_id, llm_config, research_papers_data):
    # load research papers data
    papers = research_papers_data

    # create papers dataset
    papers_left_dataset = Dataset(
        name="Papers Dataset (Left)",
        annotation="A dataset containing research papers.",
        items=papers,
    )
    papers_right_dataset = Dataset(
        name="Papers Dataset (Right)",
        annotation="A dataset containing research papers.",
        items=papers,
    )

    # create input to semantic join operator
    task = "The two papers share at least three authors in common"
    input_datasets = {papers_left_dataset.name: papers_left_dataset, papers_right_dataset.name: papers_right_dataset}

    # execute the operator
    sem_join_operator = SemJoinOperator(task, test_model_id, llm_config, "output-dataset-id", max_workers=4)
    output_datasets = sem_join_operator(papers_left_dataset.name, papers_right_dataset.name, input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 3
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 2
