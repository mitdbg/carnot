from carnot.data.dataset import Dataset
from carnot.operators.sem_groupby import SemGroupByOperator


def test_sem_groupby_operator_sem_group_relational_agg(test_model_id, llm_config):
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

    # create input to group by operator
    task = "Count the number of animals in each classification group ([mammal, bird, reptile, fish, amphibian])"
    group_by_fields = [{"name": "animal_group", "type": str, "description": "The classification group of the animal."}]
    agg_fields = [{"name": "count", "type": int, "description": "The count of animals in the group.", "func": "count"}]
    input_datasets = {animal_dataset.name: animal_dataset}

    # execute the operator
    sem_groupby_operator = SemGroupByOperator(task, group_by_fields, agg_fields, "output-dataset-id", test_model_id, llm_config, max_workers=4)
    output_datasets = sem_groupby_operator("Animal Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 4
    assert {"animal_group": "mammal", "count": 2} in output_dataset.items
    assert {"animal_group": "reptile", "count": 1} in output_dataset.items
    assert {"animal_group": "fish", "count": 1} in output_dataset.items
    assert {"animal_group": "bird", "count": 1} in output_dataset.items


def test_sem_groupby_operator_sem_group_sem_agg(test_model_id, llm_config):
    # construct dataset of various animals
    animal_data = [
        {"animal": animal, "name": name, "weight": weight}
        for animal, name, weight in [
            ("giraffe", "larry", "1,192kg"),
            ("anaconda", "annie", "7kg"),
            ("salmon", "sally", "4.5kg"),
            ("elephant", "dumbo", "6,000kg"),
            ("tucan", "sam", "0.5kg"),
        ]
    ]
    animal_dataset = Dataset(
        name="Animal Dataset",
        annotation="A dataset containing different animals.",
        items=animal_data,
    )

    # create input to group by operator
    task = "Count the number of animals in each classification group ([mammal, bird, reptile, fish, amphibian]) and report the heaviest animal's name in each group."
    group_by_fields = [{"name": "animal_group", "type": str, "description": "The classification group of the animal."}]
    agg_fields = [
        {"name": "count", "type": int, "description": "The count of animals in the group.", "func": "count"},
        {"name": "heaviest_animal", "type": str, "description": "The name of the heaviest animal in the group.", "func": "heaviest_animal"},
    ]
    input_datasets = {animal_dataset.name: animal_dataset}

    # execute the operator
    sem_groupby_operator = SemGroupByOperator(task, group_by_fields, agg_fields, "output-dataset-id", test_model_id, llm_config, max_workers=4)
    output_datasets = sem_groupby_operator("Animal Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 4
    assert {"animal_group": "mammal", "count": 2, "heaviest_animal": "dumbo"} in output_dataset.items
    assert {"animal_group": "reptile", "count": 1, "heaviest_animal": "annie"} in output_dataset.items
    assert {"animal_group": "fish", "count": 1, "heaviest_animal": "sally"} in output_dataset.items
    assert {"animal_group": "bird", "count": 1, "heaviest_animal": "sam"} in output_dataset.items


def test_sem_groupby_operator_relational_group_sem_agg(test_model_id, llm_config):
    # construct dataset of various animals
    animal_data = [
        {"animal": animal, "name": name, "weight": weight}
        for animal, name, weight in [
            ("giraffe", "larry", "1,192kg"),
            ("anaconda", "annie", "7kg"),
            ("salmon", "sally", "4.5kg"),
            ("giraffe", "barry", "1,000kg"),
            ("tucan", "sam", "0.5kg"),
        ]
    ]
    animal_dataset = Dataset(
        name="Animal Dataset",
        annotation="A dataset containing different animals.",
        items=animal_data,
    )

    # create input to group by operator
    task = "group the animals by type and report the heaviest animal's weight for each group."
    group_by_fields = [{"name": "animal", "type": str, "description": "The animal's name."}]
    agg_fields = [
        {"name": "heaviest_weight", "type": float, "description": "The weight of the heaviest animal in kg.", "func": "heaviest_weight"},
    ]
    input_datasets = {animal_dataset.name: animal_dataset}

    # execute the operator
    sem_groupby_operator = SemGroupByOperator(task, group_by_fields, agg_fields, "output-dataset-id", test_model_id, llm_config, max_workers=4)
    output_datasets = sem_groupby_operator("Animal Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 4
    assert {"animal": "giraffe", "heaviest_weight": 1192.0} in output_dataset.items
    assert {"animal": "anaconda", "heaviest_weight": 7.0} in output_dataset.items
    assert {"animal": "salmon", "heaviest_weight": 4.5} in output_dataset.items
    assert {"animal": "tucan", "heaviest_weight": 0.5} in output_dataset.items


def test_sem_groupby_operator_relational_group_relational_agg(test_model_id, llm_config):
    # construct dataset of various animals
    animal_data = [
        {"animal": animal, "name": name, "weight": weight}
        for animal, name, weight in [
            ("giraffe", "larry", 1192.0),
            ("anaconda", "annie", 7.0),
            ("salmon", "sally", 4.5),
            ("giraffe", "barry", 1000.0),
            ("tucan", "sam", 0.5),
        ]
    ]
    animal_dataset = Dataset(
        name="Animal Dataset",
        annotation="A dataset containing different animals.",
        items=animal_data,
    )

    # create input to group by operator
    task = "group the animals by type and report the heaviest animal's weight for each group."
    group_by_fields = [{"name": "animal", "type": str, "description": "The animal's name."}]
    agg_fields = [
        {"name": "weight", "type": float, "description": "The weight of the heaviest animal in kg.", "func": "max"},
    ]
    input_datasets = {animal_dataset.name: animal_dataset}

    # execute the operator
    sem_groupby_operator = SemGroupByOperator(task, group_by_fields, agg_fields, "output-dataset-id", test_model_id, llm_config, max_workers=4)
    output_datasets = sem_groupby_operator("Animal Dataset", input_datasets)

    # assert the output is as expected
    assert len(output_datasets) == 2
    assert "output-dataset-id" in output_datasets
    output_dataset = output_datasets["output-dataset-id"]
    assert len(output_dataset.items) == 4
    assert {"animal": "giraffe", "weight": 1192.0} in output_dataset.items
    assert {"animal": "anaconda", "weight": 7.0} in output_dataset.items
    assert {"animal": "salmon", "weight": 4.5} in output_dataset.items
    assert {"animal": "tucan", "weight": 0.5} in output_dataset.items
