# code for MAP@k calculation

from typing import List, Any

import numpy as np


def precision_k(
        actual: List[Any],
        predicted: List[Any],
        k: int = 6
) -> float:
    predicted = predicted[:k]

    if not actual:
        return 0.0

    act_set = set(actual)
    pred_set = set(predicted)
    result = len(act_set & pred_set) / float(k)
    return result


def recall_k(
        actual: List[Any],
        predicted: List[Any],
        k: int = 6
) -> float:
    predicted = predicted[:k]

    if not actual:
        return 0.0

    act_set = set(actual)
    pred_set = set(predicted)
    result = len(act_set & pred_set) / len(act_set)
    return result
    

def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0
    
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for ind, pred in enumerate(predicted):
        if pred in actual and pred not in predicted[:ind]:
            num_hits += 1.0
            score += num_hits / (ind + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(act, pred, k) for act, pred in zip(actual, predicted)])


def mapk_drop_empty_actual(actual, predicted, k=5):
    """
    Computes the mean average precision at k. Drop emtpy actuals 
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(act, pred, k) for act, pred in zip(actual, predicted) if act])
