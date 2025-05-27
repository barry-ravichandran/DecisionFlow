import os
import json
from collections import defaultdict, Counter


# Determine which of the two labels has a higher score and return its index (0 or 1)
def get_label_index_with_higher_score(label_list):
    score_0 = list(label_list[0].values())[0]
    score_1 = list(label_list[1].values())[0]
    return 0 if score_0 >= score_1 else 1


# Evaluate a single model run and compute accuracy metrics
def single_evaluate(detailed_infor, input_output, alignment, log_file=None):
    in_out_labels = []  # To store incorrectly predicted examples
    label_stats = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )  # Track per-label statistics

    right = 0  # Number of correct predictions
    count = 0  # Total number of evaluated samples

    for num in range(len(input_output)):
        label = input_output[num]["label"]
        label_index = get_label_index_with_higher_score(label)

        # Reverse label index if alignment is "low"
        if alignment == "low":
            label_index = -label_index + 1

        label_name = list(label[label_index].keys())[0]
        label_stats[label_name]["total"] += 1
        count += 1

        # Skip sample if it contains an error
        try:
            error = detailed_infor[num]["detailed_infor"]["error"]
            continue
        except:
            pass

        predicted_index = input_output[num]["output"]["choice"]

        # Update correct count or store incorrect case
        if label_index == predicted_index:
            right += 1
            label_stats[label_name]["correct"] += 1
        else:
            in_out_labels.append(
                {
                    "input": detailed_infor[num]["input"],
                    "label": label,
                    "output": input_output[num]["output"]["info"]["reasoning"],
                }
            )

    # Logging function that writes to file or prints to console
    def log(msg):
        if log_file:
            log_file.write(msg + "\n")
        else:
            print(msg)

    # Handle special case for "unaligned" alignment evaluation
    if alignment == "unaligned":
        log(
            f"Unaligned High: {right / count if count != 0 else 0:.4f}, Total samples: {count}\n"
            f"Unaligned Low: {1 - right / count if count != 0 else 0:.4f}, Total samples: {count}"
        )
        log("Per-label stats:")
        for label, stats in label_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0
            log(f"  {label} -> Total: {total}, High: {correct}, High rate: {acc:.2f}")
    else:
        # Standard evaluation summary
        log(
            f"Overall Accuracy: {right / count if count != 0 else 0:.4f}, Total samples: {count}"
        )
        log("Per-label stats:")
        for label, stats in label_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0
            log(f"  {label} -> Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}")

    return in_out_labels  # Return the list of incorrect predictions


# Main evaluation function that loads prediction results, evaluates them, and writes logs and error cases
def medical_evaluate(results_dir, out_dir, alignment):
    path = os.path.join("DecisionFlow_results", "MTA", results_dir)

    # Loop over all models in the results directory (e.g., qwen-2.5-7b)
    for model in os.listdir(path):
        model_path = os.path.join(path, model)
        if not os.path.isdir(model_path):
            continue

        # Loop over all alignment subfolders (e.g., 'align', 'unaligned')
        for alignment_ in os.listdir(model_path):
            print(alignment_)  # Print current alignment subdir
            alignment_path = os.path.join(model_path, alignment_)
            if not os.path.isdir(alignment_path):
                continue

            # Construct paths to input files
            detailed_infor_path = os.path.join(alignment_path, "detailed_infor.json")
            print(results_dir, model, alignment)  # Debug output
            input_output_path = os.path.join(alignment_path, "input_output_labels.json")
            change_path = os.path.join(alignment_path, "error_cases.json")

            # Prepare output directory and log file path
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, f"{results_dir}_{model}_{alignment}.txt")

            # Run evaluation if input files exist
            if os.path.exists(detailed_infor_path):
                with open(detailed_infor_path, "r") as f:
                    detailed_infor = json.load(f)
                with open(input_output_path, "r") as f:
                    input_output = json.load(f)

                # Write evaluation log to file
                with open(log_path, "w") as log_file:
                    in_out_labels = single_evaluate(
                        detailed_infor,
                        input_output,
                        alignment,
                        log_file,
                    )

                # Save incorrectly predicted cases for further inspection
                with open(change_path, "w") as f:
                    json.dump(in_out_labels, f, indent=4)
            else:
                print(f"File not found: {detailed_infor_path}")
