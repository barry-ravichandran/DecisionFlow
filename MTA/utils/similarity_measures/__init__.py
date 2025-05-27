from typing import List
from functools import partial


def force_choice(similarity_measure_func, text: str, choices: List[str]):
    top_score = -float('inf')
    top_choice = None
    top_choice_idx = None
    for i, choice in enumerate(choices):
        score = similarity_measure_func(text, choice)

        # Assumes higher score is better match
        # TODO: Add option to prioritize lower score instead of higher
        if score > top_score:
            top_score = score
            top_choice = choice
            top_choice_idx = i

    return top_choice_idx, top_choice


def get_similarity_measure_func(measure_name):
    if measure_name == "bert":
        from DecisionFlow.DecisionFlow.MTA.utils.similarity_measures.bert_score import (
            bert_score_similarity_f1)
        similarity_measure_func = bert_score_similarity_f1
    elif measure_name == "heuristic":
        from DecisionFlow.DecisionFlow.MTA.utils.similarity_measures.heuristics import (
            score_string_similarity,
        )

        similarity_measure_func = score_string_similarity
    else:
        raise NotImplementedError("Unrecognized similarity measure '{}', "
                                  "aborting!".format(measure_name))

    return similarity_measure_func


def build_force_choice_func(measure_name):
    similarity_measure = get_similarity_measure_func(measure_name)

    return partial(force_choice, similarity_measure)
