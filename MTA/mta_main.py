import json
import os
from datetime import datetime
from MTA.scripts.mta_generate import generate_outputs


def mta_function(
    method, model, model_path, temperature, results_path, dataset_name, alignment, repeat=False
):

    with open(os.path.join("MTA", 'data', dataset_name), "r") as f:
        dataset = json.load(f)
        dataset = dataset

    experiment_name = model
    if method == "self-consistency":
        for i in range(3):
            print("Repeat:", i)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            results_basename = f"{experiment_name}__{timestamp}"
            results_dir = os.path.join("DecisionFlow_results", "MTA", results_path, results_basename, alignment)
            os.makedirs(results_dir, exist_ok=True)

            generated_outputs, detailed = generate_outputs(
                dataset=dataset,
                method=method,
                model=model,
                model_path=model_path,
                results_dir=results_dir,
                alignment=alignment,
                temperature=temperature,
            )

            in_out_labels = []
            for generated_output, (input_, label) in zip(generated_outputs, dataset):
                in_out_labels.append(
                    {
                        "input": input_,
                        "label": label,
                        "output": generated_output,
                    }
                )
            detailed_infor = []
            for detail, (input_, label) in zip(detailed, dataset):
                detailed_infor.append(
                    {
                        "input": input_,
                        "label": label,
                        "detailed_infor": detail,
                    }
                )

            with open(os.path.join(results_dir, "input_output_labels.json"), "w") as f:
                json.dump(in_out_labels, f, indent=4)
            with open(os.path.join(results_dir, "detailed_infor.json"), "w") as f:
                json.dump(detailed_infor, f, indent=4)
    else:
        if repeat:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            results_basename = f"{experiment_name}__{timestamp}"
        else:
            results_basename = experiment_name

        results_dir = os.path.join("DecisionFlow_results", "MTA", results_path, results_basename, alignment)
        os.makedirs(results_dir, exist_ok=True)

        generated_outputs, detailed = generate_outputs(
            dataset=dataset,
            method=method,
            model=model,
            model_path=model_path,
            results_dir=results_dir,
            alignment=alignment,
            temperature=temperature,
        )

        in_out_labels = []
        for generated_output, (input_, label) in zip(generated_outputs, dataset):
            in_out_labels.append(
                {
                    "input": input_,
                    "label": label,
                    "output": generated_output,
                }
            )
        detailed_infor = []
        for detail, (input_, label) in zip(detailed, dataset):
            detailed_infor.append(
                {
                    "input": input_,
                    "label": label,
                    "detailed_infor": detail,
                }
            )

        with open(os.path.join(results_dir, "input_output_labels.json"), "w") as f:
            json.dump(in_out_labels, f, indent=4)
        with open(os.path.join(results_dir, "detailed_infor.json"), "w") as f:
            json.dump(detailed_infor, f, indent=4)
