"""Evaluation functions for retrieval systems."""

from typing import Set, Iterable, Union


def recall(predicted: Union[Set[str], Iterable[str]], gold: Union[Set[str], Iterable[str]]) -> float:
    """
    Compute recall accuracy for retrieval.
    
    Recall is defined as: |predicted ∩ gold| / |gold|
    
    Args:
        predicted: Set or iterable of predicted document IDs
        gold: Set or iterable of gold/relevant document IDs
        
    Returns:
        Recall score as a float between 0.0 and 1.0
    """
    predicted_set = set(predicted) if not isinstance(predicted, set) else predicted
    gold_set = set(gold) if not isinstance(gold, set) else gold
    
    if len(gold_set) == 0:
        raise ValueError("Gold set is empty")
    if len(predicted_set) == 0:
        raise ValueError("Predicted set is empty")
    
    covered_docs = gold_set.intersection(predicted_set)
    return len(covered_docs) / len(gold_set)


