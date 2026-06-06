"""
Run train_eval_pose.py for several model / hyper-parameter configurations
declared in a YAML experiment file.

The YAML file may define:
  - defaults: parameters applied to every experiment unless overridden,
  - global keys (data_config, out_dir, runs_dir) shared by all experiments,
  - experiments: an explicit list of configurations,
  - grid: parameter lists expanded into their full Cartesian product.

Example
-------
python run_experiments.py --config experiments.yaml --continue-on-error --compare
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

GLOBAL_KEYS = ("data_config", "out_dir", "runs_dir")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="models/evaluation/experiments.yaml",
                        help="YAML file describing the experiments to run.")
    parser.add_argument("--script", default="models/evaluation/train_eval_pose.py",
                        help="Path to the training/evaluation script.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter used to launch each run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep going if one experiment fails.")
    parser.add_argument("--compare", action="store_true",
                        help="Run compare_pose_results.py once all runs finish.")
    parser.add_argument("--compare-out", default="comparison",
                        help="Output directory for the comparison step.")
    return parser.parse_args()


def expand_experiments(config):
    """Merge defaults and global keys into every explicit and grid experiment."""
    defaults = config.get("defaults", {})
    shared = {key: config[key] for key in GLOBAL_KEYS if key in config}
    experiments = []

    for experiment in config.get("experiments", []):
        experiments.append({**defaults, **shared, **experiment})

    grid = config.get("grid")
    if grid:
        keys = list(grid.keys())
        for combination in itertools.product(*(grid[key] for key in keys)):
            overrides = dict(zip(keys, combination))
            experiments.append({**defaults, **shared, **overrides})

    return experiments


def build_command(python_exe, script, params):
    """Translate a parameter dict into a command line for the training script."""
    command = [python_exe, script]
    for key, value in params.items():
        if value is None:
            continue
        command.append(f"--{key.replace('_', '-')}")
        command.append(str(value))
    return command


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiments = expand_experiments(config)

    if not experiments:
        print("No experiment found in the configuration file.")
        return

    print(f"{len(experiments)} experiment(s) to run")
    failures = []

    for index, params in enumerate(experiments, start=1):
        command = build_command(args.python, args.script, params)
        print(f"\n[{index}/{len(experiments)}] {' '.join(command)}")
        if args.dry_run:
            continue

        result = subprocess.run(command)
        if result.returncode != 0:
            failures.append((index, params.get("model", "unknown")))
            print(f"Experiment {index} failed (exit code {result.returncode}).")
            if not args.continue_on_error:
                sys.exit(result.returncode)

    if failures:
        print("\nFailed experiments:")
        for index, model in failures:
            print(f"  - [{index}] {model}")

    if args.compare and not args.dry_run:
        results_dir = config.get("out_dir", "pose_results")
        compare_command = [
            args.python, "models/evaluation/compare_pose_results.py",
            "--results-dir", results_dir,
            "--out-dir", args.compare_out,
        ]
        print(f"\nRunning comparison: {' '.join(compare_command)}")
        subprocess.run(compare_command)

    print("\nAll experiments processed.")


if __name__ == "__main__":
    main()
