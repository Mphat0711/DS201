import argparse
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the Lab 4 translation experiments")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=("all", "none", "bahdanau", "luong"), default="all")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the selected checkpoint on test")
    args = parser.parse_args()
    folders = {"none": "assignment_1_no_attention", "bahdanau": "assignment_2_bahdanau", "luong": "assignment_3_luong"}
    names = list(folders) if args.model == "all" else [args.model]
    results = [run_experiment(name, args.data_dir, args.output_dir / folders[name], ExperimentConfig(), args.evaluate_test) for name in names]
    if args.model == "all":
        rows = [{"model": r.name, "best_epoch": r.best_epoch, "dev_rougeL_f1": r.dev_metrics["rougeL_f1"],
                 "test_rougeL_f1": r.test_metrics["rougeL_f1"] if r.test_metrics else None,
                 "parameter_count": r.parameter_count, "seconds_per_epoch": r.seconds_per_epoch} for r in results]
        pd.DataFrame(rows).to_csv(args.output_dir / "comparison.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
