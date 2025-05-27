import argparse
import traceback
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument(
        "--action", type=str, default="inference", choices=["inference", "evaluation"]
    )
    parser.add_argument(
        "--dataset", type=str, default="mta", choices=["mta", "agriculture", "stocks"]
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0
    )
    parser.add_argument(
        "--model",
        type=str,
        default="closed_source",
        choices=["open_source", "closed_source"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )

    # Medical Dataset
    parser.add_argument('--mta_method',
                        type=str,
                        default='decisionflow',
                        choices=['decisionflow', 'zero-shot', 'cot', 'self-consistency']
                        )
    parser.add_argument('--mta_alignment',
                        type=str,
                        default='high',
                        choices=['high', 'low', 'unaligned']
                        )
    parser.add_argument('--mta_dataset',
                        type=str,
                        default='MTA_data.json',
                        )
    parser.add_argument('--mta_infer_path',
                        type=str,
                        default='mta_results',
                        help='Path to which need to be evaluated'
                        )
    parser.add_argument('--mta_eval_path',
                        type=str,
                        default='mta_results',
                        help='Path to which need to be evaluated'
                        )
    parser.add_argument('--mta_eval_output_path',
                        default="DecisionFlow_results/evaluate_result",
                        type=str,
                        help='Evaluation result output path'
                        )

    # DeLLMa
    parser.add_argument("--year", type=str, default="2021")
    parser.add_argument(
        "--sample_size", type=int, default=16, help="number of beliefs to sample"
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=32,
        help="minibatch size for DeLLMa prompt",
    )
    parser.add_argument(
        "--overlap_pct",
        type=float,
        default=0.25,
        help="overlap percentage for DeLLMa prompt",
    )
    parser.add_argument(
        "--sc-samples",
        type=int,
        default=5,
        help="number of samples for self-consistency",
    )
    parser.add_argument(
        "--dellma_infer_path", type=str, default="dellma_results", help="path to data folder for dellma"
    )

    parser.add_argument(
        "--dellma_eval_path",
        type=str,
        default="dellma_results",
        help="path to data folder",
    )

    # Method
    parser.add_argument(
        "--dellma_mode",
        type=str,
        default="zero-shot",
        choices=["decisionflow", "zero-shot", "self-consistency", "cot", "rank", "rank-minibatch"],
    )
    parser.add_argument(
        "--dellma_eval_mode",
        type=str,
        default="top1",
        choices=[
            "top1",
            "pairwise",
        ],
    )
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha for ILSR")
    args = parser.parse_args()

    # Run inference
    if args.action == "inference":
        if args.dataset == "mta":
            from MTA import mta_main
            mta_main.mta_function(
                args.mta_method,
                args.model,
                args.model_path,
                args.temperature,
                args.mta_infer_path,
                args.mta_dataset,
                args.mta_alignment,
            )
        elif args.dataset == "agriculture" or args.dataset == "stocks":
            from DeLLMa import dellma_main
            try:
                dellma_main.agriculture_stocks_function(
                    args.dataset,
                    args.year,
                    args.dellma_infer_path,
                    args.dellma_mode,
                    args.model,
                    args.model_path,
                    args.temperature
                )
            except Exception as e:
                print(e)
                traceback.print_exc()  # Print the full stack trace

    # Run evaluation
    else:
        if args.dataset == "mta":
            if args.mta_method == "self-consistency":
                from MTA.mta_evaluate_sc import medical_evaluate
                medical_evaluate(
                    results_dir=args.mta_eval_path,
                    out_dir=args.mta_eval_output_path,
                    alignment=args.mta_alignment,
                )
            else:
                from MTA.mta_evaluate import medical_evaluate
                medical_evaluate(
                    results_dir=args.mta_eval_path,
                    out_dir=args.mta_eval_output_path,
                    alignment=args.mta_alignment,
                )
        elif args.dataset == "agriculture" or args.dataset == "stocks":
            if args.dataset == "agriculture":
                agent_name = "farmer"
            elif args.dataset == "stocks":
                agent_name = "trader"
            from DeLLMa.dellma_evaluate import evaluate_dellma
            evaluate_dellma(
                agent_name,
                args.year,
                args.dellma_eval_path,
                args.dellma_mode,
                args.sample_size,
                args.minibatch_size,
                args.overlap_pct,
                args.alpha,
                args.dellma_eval_mode,
            )
