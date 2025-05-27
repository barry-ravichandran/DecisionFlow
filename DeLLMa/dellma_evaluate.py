import os
import json
import choix
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from typing import List, Tuple, Dict
from DeLLMa.utils.data_utils import get_combinations, AGNOSTIC_STATES
from dataclasses import dataclass, field
from scipy.special import softmax

@dataclass
class PreferenceConfig:
    pref_enum_mode: str = "base"
    sample_size: int = 50  # number of states to be sampled from the belief distribution
    minibatch_size: int = 50  # size of the minimbatch
    overlap_pct: float = 0.2  # percentage of overlap between minibatches

# Define belief to score mapping
belief2score: Dict[str, float] = {
    "very likely": 6,
    "likely": 5,
    "somewhat likely": 4,
    "somewhat unlikely": 3,
    "unlikely": 2,
    "very unlikely": 1,
}


def setup_output_dirs(agent_name: str, year: str, pref_enum_mode: str):
    """Setup directories for output logging"""
    base_dir = f"prediction_logs/{agent_name}_{year}_{pref_enum_mode}"
    os.makedirs(base_dir, exist_ok=True)
    correct_dir = os.path.join(base_dir, "correct")
    error_dir = os.path.join(base_dir, "errors")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    return correct_dir, error_dir


def save_prediction_result(dir_path: str, choices: List[str], data: dict):
    """Save prediction result to JSON file"""
    filename = f"pred_{'-'.join(choices)}.json"
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_state_beliefs(agent_name, year="2021"):
    if agent_name == "farmer":
        state_beliefs = json.load(open(f"cache/{agent_name}_{year}_states.json"))
    else:
        state_beliefs = json.load(open(f"cache/{agent_name}_states.json"))
    state2val2prob = {
        state: {val: 0.0 for val in val2belief.keys()}
        for state, val2belief in state_beliefs.items()
    }
    for state, val2belief in state_beliefs.items():
        total_score = sum([belief2score[v] for v in val2belief.values()])
        for val, belief in val2belief.items():
            state2val2prob[state][val] = belief2score[belief] / total_score
    return state2val2prob


def get_state_value_prob(state_value, state2val2prob):
    prob = 1.0
    for s, v in state_value:
        prob *= state2val2prob[s][v]
    return prob


def parse_number(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("".join(s.split(",")))


def get_stock_optimal_action(stocks, prediction: int):
    source_date = "2023-12-01"
    target_date = "2023-12-29"
    ground_truth = []
    predicted_utility = 0.0

    for choice_idx, stock in enumerate(stocks):
        stock_df = pd.read_csv(f"DeLLMa/data/stocks/{stock.upper()}.csv")
        stock_open = stock_df[stock_df.Date == source_date]["Open"].values[0]
        stock_close = stock_df[stock_df.Date == target_date]["Close"].values[0]
        ground_truth.append(stock_close / stock_open)
        if choice_idx == prediction:
            predicted_utility = stock_close / stock_open

    optimal_action = ground_truth.index(max(ground_truth))
    return optimal_action, max(ground_truth) - 1, predicted_utility - 1


def get_agriculture_optimal_action(fruits: List[str], prediction: int, year: str):
    next_year = str(int(year) + 1)
    df = pd.read_csv(
        f"DeLLMa/data/agriculture/stats/CA-{next_year}.csv"
    )
    ground_truth = []
    predicted_utility = 0.0

    for choice_idx, fruit in enumerate(fruits):
        unit_yield = parse_number(
            df[df["Commodity"] == fruit]["Yield"].item().split()[0]
        )
        unit_price = parse_number(
            df[df["Commodity"] == fruit]["Price per Unit"].item().split()[0]
        )
        utility = unit_yield * unit_price
        ground_truth.append(utility)

        if choice_idx == prediction:
            predicted_utility = utility

    optimal_action = ground_truth.index(max(ground_truth))
    return optimal_action, max(ground_truth), predicted_utility


def parse_base_response(
    choices: List[str],
    agent_name: str,
    year: str = "2021",
    results_path: str = "data",
    pref_enum_mode: str = "zero-shot",
    error_dir: str = None,
):
    if agent_name == "farmer":
        domain_path = f"agriculture/{year}"
    else:
        domain_path = "stocks"
    result_path = (
        f"DeLLMa_results/{results_path}/{domain_path}/{pref_enum_mode}/{('-'.join(choices))}"
    )
    try:
        if (
            pref_enum_mode == "zero-shot"
            or pref_enum_mode == "self-consistency"
            or pref_enum_mode == "decisionflow"
        ):
            response = json.load(
                open(os.path.join(result_path, f"response/response.json"))
            )
        elif pref_enum_mode == "cot":
            response = json.load(
                open(os.path.join(result_path, f"response/response.json"))
            )
        decision = response["decision"]
        action_name = decision.split(":")[0].split(".")[1].strip()
        pred = int(decision.split(".")[0].split()[1]) - 1
        return pred, action_name
    except Exception as e:
        error_data = {
            "choices": choices,
            "error": str(e),
            "result_path": result_path,
            "response": response if "response" in locals() else None,
        }
        if error_dir:
            save_prediction_result(error_dir, choices, error_data)
        raise ValueError(f"Error processing {choices}: {str(e)}")


def predict_one_sample(
    choices: List[str],
    preference_config: PreferenceConfig,
    agent_name: str,
    alpha: float = 0.01,
    mode: str = "mc",
    softmax_mode: str = "full",
    temperature: float = 1,
    year: str = "2021",
    results_path: str = "data",
) -> Tuple[int, List[float]]:
    """Utility Elicitation from pairwise preferences or ranked preferences.
    Args:
        preference: List of pairwise preferences or ranked preferences.
        preference_type: "zero-shot", "self-consitency", "cot", "rank", or "rank-minibatch".
    """
    if preference_config.pref_enum_mode in [
        "zero-shot",
        "cot",
        "self-consistency",
        "decisionflow",
    ]:
        pred, action_name = parse_base_response(
            choices,
            agent_name,
            year=year,
            results_path=results_path,
            pref_enum_mode=preference_config.pref_enum_mode,
        )
        utilities = [0] * len(choices)
        utilities[pred] = 1
        return (pred, action_name), utilities

    state_values, actions, comparison_pairs, mapped_ranks = parse_rank_prompt_response(
        choices=choices,
        preference_config=preference_config,
        agent_name=agent_name,
        year=year,
        results_path=results_path,
    )
    if "minibatch" in preference_config.pref_enum_mode:
        data_size = len(choices) * preference_config.sample_size
    if preference_config.pref_enum_mode == "rank":
        data_size = preference_config.sample_size
    if mode == "pairwise":
        util_fct = partial(
            choix.ilsr_pairwise,
            n_items=data_size,
            alpha=alpha,
            max_iter=10_000,
        )
        data = comparison_pairs
    elif mode == "top1":
        util_fct = partial(
            choix.ilsr_top1,
            n_items=data_size,
            alpha=alpha,
            max_iter=10_000,
        )
        data = []
        for rank in mapped_ranks:
            if not rank:
                continue
            data.append([rank[0], [rank[i] for i in range(1, len(rank))]])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scores = util_fct(data=data)

    if softmax_mode == "full":
        if temperature == 0:
            scores = softmax(scores)
        else:
            scores = softmax(scores / temperature)
    elif softmax_mode == "action":
        for a in sorted(set(actions)):
            if temperature == 0:
                scores[actions == a] = softmax(scores[actions == a])
            else:
                scores[actions == a] = softmax(scores[actions == a] / temperature)
    utilities = [
        scores[actions == a].sum() / (actions == a).sum() for a in sorted(set(actions))
    ]

    pred = np.argmax(utilities)
    return pred, utilities


def parse_rank_prompt_response(
    choices: List[str],
    preference_config: PreferenceConfig,
    agent_name: str,
    year: str = "2021",
    results_path: str = "data",
) -> Tuple[List[List[Tuple[str, str]]], np.ndarray, List[Tuple[int, int]]]:
    if agent_name not in ["farmer", "trader"]:
        raise ValueError("agent_name must be either 'farmer' or 'trader'")

    if agent_name == "farmer":
        domain_path = f"agriculture/{year}"
    else:
        domain_path = "stocks"

    if preference_config.pref_enum_mode in ["rank", "rank-minibatch", "decisionflow"]:
        if "minibatch" in preference_config.pref_enum_mode:
            config_path = (
                f"sample_size={preference_config.sample_size}_"
                f"minibatch_size={preference_config.minibatch_size}_"
                f"overlap_pct={int(preference_config.overlap_pct*100)}"
            )
        else:
            config_path = ""
        result_path = f"DeLLMa_results/{results_path}/{domain_path}/{preference_config.pref_enum_mode}/{'-'.join(choices)}"
    else:
        raise NotImplementedError
    prompt_path = os.path.join(result_path, "prompt")
    response_path = os.path.join(result_path, "response")
    action_size = len(choices)

    if "minibatch" or "math_model" in preference_config.pref_enum_mode:
        data_size = action_size * preference_config.sample_size
    else:
        data_size = preference_config.sample_size

    state_values = [[] for _ in range(data_size)]
    added_state_values = [False] * data_size

    actions = [-1] * data_size
    comparison_pairs = []
    mapped_ranks = []

    for response_idx in range(len(os.listdir(response_path))):
        prompt_fname = os.path.join(prompt_path, f"prompt_{response_idx}.txt")
        response_fname = os.path.join(response_path, f"response_{response_idx}.json")

        if not os.path.exists(prompt_fname):
            continue
        prompt_lines = open(prompt_fname).readlines()
        state_action_strings = list(
            filter(lambda x: x.startswith("- State-Action Pair"), prompt_lines)
        )
        if len(state_action_strings) != preference_config.minibatch_size:
        #     # os.remove(prompt_fname)
        #     # os.remove(response_fname)
            # print(f"Removed invalid prompt/response pair: {response_idx}")
            continue

        # load and post-process rank
        try:
            response = json.loads(open(response_fname).read())
            if "rank" in response:
                rank = response["rank"]
            else:
                assert "order" in response
                rank = response["order"]
        except:
            print("error reading response file", response_fname)
            rank = None
        if not rank:
            # sometimes gpt4 refuses to answer
            rank = []

        # pad rank if necessary
        rank = [int(r) - 1 for r in rank]
        if set(rank) != set(range(len(rank))):
            rank = rank + list(set(range(len(rank))) - set(rank))

        prompt = open(prompt_fname).readlines()
        state_action_strings = list(
            filter(
                lambda x: x.startswith("- State-Action Pair"),
                prompt,
            )
        )
        action_map = extract_action_map_from_prompt(prompt_fname)

        # map index within state-action pair to index in the full batch
        index_offset = response_idx * int(
            preference_config.overlap_pct * preference_config.minibatch_size
        )
        mbidx2dataidx = []

        for minibatch_idx, state_action_string in enumerate(state_action_strings):
            data_idx = (
                response_idx * preference_config.minibatch_size
                + minibatch_idx
                - index_offset
            )
            mbidx2dataidx.append(data_idx)
            # if data_idx >= len(state_values):
            #     continue
            if state_values[data_idx]:
                continue
            buffer = ""
            for state_value in (
                state_action_string.split("; Action: ")[0].split("State: ")[1].split(",")
            ):
                if len(state_value.split(": ")) != 2:
                    buffer += f"{state_value},"
                    continue
                else:
                    if buffer != "":
                        state_value = buffer + state_value
                        buffer = ""
                state, value = state_value.split(": ")
                state = state.strip()
                value = value.strip()
                choice_in_state = False
                if state in AGNOSTIC_STATES:
                    choice_in_state = True
                for choice in choices:
                    if choice.lower() in state.lower():
                        choice_in_state = True
                if not choice_in_state:
                    continue
                state_values[data_idx].append((state, value))
            action_str = state_action_string.split("; Action: ")[1].strip().lower()
            if action_str not in action_map:
                raise ValueError(f"Unknown action string: '{action_str}'")
            actions[data_idx] = action_map[action_str]

        mapped_ranks.append([mbidx2dataidx[r] for r in rank if r < len(mbidx2dataidx)])

        for i, ri in enumerate(rank):
            for j, rj in enumerate(rank[i + 1 :]):
                if ri >= len(mbidx2dataidx) or rj >= len(mbidx2dataidx):
                    continue
                comparison_pairs.append((mbidx2dataidx[ri], mbidx2dataidx[rj]))

    return (
        state_values,
        np.array(actions).astype(int),
        comparison_pairs,
        mapped_ranks,
    )


def extract_action_map_from_prompt(prompt_path: str) -> Dict[str, int]:
    with open(prompt_path, "r") as f:
        lines = f.readlines()

    action_map = {}
    action_pattern = re.compile(r"Action\s+(\d+)\.\s+(.+)")
    for line in lines:
        match = action_pattern.match(line.strip())
        if match:
            idx = int(match.group(1)) - 1  # zero-based index
            action_str = match.group(2).strip().lower()
            action_map[action_str] = idx
    return action_map


def predict(
    preference_config: PreferenceConfig,
    agent_name: str,
    alpha: float = 0.01,
    mode: str = "mc",
    softmax_mode: str = "full",
    temperature: float = 1,
    year: str = "2021",
    results_path: str = "data",
) -> Tuple[int, List[float]]:
    combs = get_combinations(agent_name, source_year=year)
    perf_by_size = defaultdict(list)
    perf_by_product = defaultdict(list)

    if agent_name == "farmer":
        optimal_action_func = partial(get_agriculture_optimal_action, year=year)
    elif agent_name == "trader":
        optimal_action_func = get_stock_optimal_action
    num = 0
    error = []
    for choices in tqdm(combs):
        # print(choices)
        (pred, action_name), utilities = predict_one_sample(
            choices=choices,
            preference_config=preference_config,
            agent_name=agent_name,
            alpha=alpha,
            mode=mode,
            softmax_mode=softmax_mode,
            temperature=temperature,
            year=year,
            results_path=results_path,
        )
        ground_truth, ground_truth_utility, predicted_utility = optimal_action_func(
            choices, pred
        )
        perf_by_size[len(choices)].append(
            (pred, ground_truth, ground_truth_utility, predicted_utility)
        )
        for product in choices:
            perf_by_product[product].append(
                (pred, ground_truth, ground_truth_utility, predicted_utility)
            )
        if pred != ground_truth:
            error.append([choices, action_name])
    return perf_by_size, perf_by_product, error


def evaluate_dellma(
    agent_name,
    year,
    results_path,
    pref_enum_mode,
    sample_size=16,
    minibatch_size=32,
    overlap_pct=0,
    alpha=2e-3,
    mode="top1",
):
    # Setup output directories
    preference_config = PreferenceConfig(
        pref_enum_mode=pref_enum_mode,
        sample_size=sample_size,
        minibatch_size=minibatch_size,
        overlap_pct=overlap_pct,
    )
    perf_by_size = defaultdict(list)
    perf_by_product = defaultdict(list)
    temperature=0
    softmax_mode="full"
    curr_perf_by_size, curr_perf_by_product, error_list = predict(
        preference_config=preference_config,
        agent_name=agent_name,
        alpha=alpha,
        mode=mode,
        softmax_mode=softmax_mode,
        temperature=temperature,
        year=year,
        results_path=results_path,
    )
    merged = []
    for key, val in curr_perf_by_size.items():
        perf_by_size[key] += val
    for key, val in curr_perf_by_product.items():
        perf_by_product[key] += val
    accs = []
    gaps = []
    for key in sorted(curr_perf_by_size.keys()):
        merged += curr_perf_by_size[key]
        accs.append(
            round(
                np.mean([pred == gt for pred, gt, _, _ in curr_perf_by_size[key]])
                * 100,
                3,
            )
        )
        gaps.append(
            round(
                np.mean(
                    [
                        pred_util / opt_util * 100
                        for _, _, opt_util, pred_util in curr_perf_by_size[key]
                    ]
                ),
                3,
            )
        )
    print("Accs", accs)
    print("All-Acc", round(np.mean([pred == gt for pred, gt, _, _ in merged]) * 100, 3))
    print("Opts (%)", gaps)
    print(
        "All-Opt (%)",
        round(
            np.mean(
                [pred_util / opt_util * 100 for _, _, opt_util, pred_util in merged]
            ),
            3,
        ),
    )
