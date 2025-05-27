import os
import json
from collections import defaultdict, Counter


# Determine which label (among two) has the higher score and return its index (0 or 1)
def get_label_index_with_higher_score(label_list):
    score_0 = list(label_list[0].values())[0]
    score_1 = list(label_list[1].values())[0]
    return 0 if score_0 >= score_1 else 1


# Evaluate a single model run: compute overall accuracy and per-label stats, log results,
# and return incorrectly predicted examples
def single_evaluate(detailed_infor, input_output, alignment, log_file=None):
    in_out_labels = []  # Store incorrectly predicted samples
    label_stats = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )  # Per-label accuracy stats

    right = 0  # Count of correct predictions
    count = 0  # Total number of evaluated samples

    for num in range(len(detailed_infor)):
        label = detailed_infor[num]["label"]
        label_index = get_label_index_with_higher_score(label)

        # If alignment is 'low', we flip the label index
        if alignment == "low":
            label_index = -label_index + 1

        label_name = list(label[label_index].keys())[0]
        label_stats[label_name]["total"] += 1
        count += 1

        # Skip samples that contain an error
        try:
            error = detailed_infor[num]["detailed_infor"]["error"]
            continue
        except:
            pass

        predicted_index = input_output[num]["output"]["choice"]

        # Compare prediction with label
        if label_index == predicted_index:
            right += 1
            label_stats[label_name]["correct"] += 1
        else:
            # Store input, label, and reasoning for incorrect prediction
            in_out_labels.append(
                {
                    "input": detailed_infor[num]["input"],
                    "label": label,
                    "output": input_output[num]["output"]["info"]["reasoning"],
                }
            )

    # Logging function: writes to file or prints
    def log(msg):
        if log_file:
            log_file.write(msg + "\n")
        else:
            print(msg)

    # Output overall accuracy
    log(
        f"Overall Accuracy: {right / count if count != 0 else 0:.4f}, Total samples: {count}"
    )
    log("Per-label stats:")

    # Output per-label statistics
    for label, stats in label_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0
        log(f"  {label} -> Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}")

    return in_out_labels


# Apply self-consistency voting over multiple inference runs
# For each sample, use majority vote to determine final predicted choice
def self_consistency_vote(all_input_outputs):
    merged_output = []
    num_samples = len(all_input_outputs[0])
    for i in range(num_samples):
        choices = []
        info_list = []
        for run in all_input_outputs:
            choices.append(run[i]["output"]["choice"])
            info_list.append(run[i]["output"]["info"])
        most_common_choice, _ = Counter(choices).most_common(1)[0]
        merged_output.append(
            {
                "output": {
                    "choice": most_common_choice,
                    "info": info_list[
                        0
                    ],  # just keep one reasoning info (could be improved)
                }
            }
        )
    return merged_output


# Perform full evaluation over a directory of model outputs using self-consistency voting
def medical_evaluate(results_dir, out_dir, alignment):
    path = os.path.join("DecisionFlow_results", "MTA", results_dir)
    all_detailed_infor = None
    all_input_outputs = []

    # Loop through models (e.g., closed_source model)
    for model in os.listdir(path):
        model_path = os.path.join(path, model)
        if not os.path.isdir(model_path):
            continue

        # Loop through alignment subdirectories (e.g., high)
        for alignment_ in os.listdir(model_path):
            alignment_path = os.path.join(model_path, alignment_)
            if not os.path.isdir(alignment_path):
                continue

            # Load result files if present
            detailed_infor_path = os.path.join(alignment_path, "detailed_infor.json")
            input_output_path = os.path.join(alignment_path, "input_output_labels.json")

            if os.path.exists(detailed_infor_path) and os.path.exists(
                input_output_path
            ):
                with open(detailed_infor_path, "r") as f:
                    detailed_infor = json.load(f)
                with open(input_output_path, "r") as f:
                    input_output = json.load(f)

                if all_detailed_infor is None:
                    all_detailed_infor = detailed_infor
                all_input_outputs.append(input_output)
            else:
                print(f"Missing files under {alignment_path}")

        if all_detailed_infor is None or len(all_input_outputs) == 0:
            print(f"No valid runs found under {alignment_path}")
            continue

        # Apply self-consistency vote across all model runs
        merged_input_output = self_consistency_vote(all_input_outputs)

        # Create output directory if not exists
        os.makedirs(out_dir, exist_ok=True)

        # Save evaluation log
        log_path = os.path.join(
            out_dir, f"{results_dir}_{model}_{alignment}_self_consistency.txt"
        )
        with open(log_path, "w") as log_file:
            in_out_labels = single_evaluate(
                all_detailed_infor,
                merged_input_output,
                alignment,
                log_file,
            )

        # Save incorrectly predicted samples
        merged_output_path = os.path.join(
            out_dir,
            f"{results_dir}_{model}_{alignment}_self_consistency.json",
        )
        with open(merged_output_path, "w") as f:
            json.dump(in_out_labels, f, indent=4)

        print(f"Self-consistency evaluation saved to {log_path}")
