import argparse
import json
import pprint
from collections import defaultdict

import os

import subprocess

from config.recognition_config import parse_config


def determine_snapshot_names(log_dir):
    snapshot_files = [file_name for file_name in os.listdir(log_dir) if os.path.splitext(file_name)[-1] == ".npz"]
    snapshots_iterations = [os.path.splitext(file_name)[0].split('_') for file_name in snapshot_files]
    snapshots, iterations = list(zip(*snapshots_iterations))
    iterations = list(sorted(set(map(int, iterations))))
    snapshot_names = set(snapshots)

    snapshot_map = {}
    for snapshot_name in snapshot_names:
        if "localizer" in snapshot_name.lower():
            snapshot_map["localizer"] = snapshot_name
        elif "recognizer" in snapshot_name.lower():
            snapshot_map["recognizer"] = snapshot_name
        else:
            snapshot_map["assessor"] = snapshot_name

    snapshot_map['iterations'] = iterations
    return snapshot_map


def find_best_result(eval_data):
    score_per_snapshot_name = {
        snapshot_name: sum([result['case_insensitive_line_accuracy'] for result in snapshot_results])
        for snapshot_name, snapshot_results in eval_data.items()
    }
    return eval_data[max(score_per_snapshot_name, key=lambda x: score_per_snapshot_name[x])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluate.py on all datasets in a given config")
    parser.add_argument("config", help="Path to config file to get datasets from")
    parser.add_argument("gpu", type=int, help="GPU Id to use")
    parser.add_argument("-i", "--iteration", default='', help="iteration to evaluate, if not given eval all snapshots")
    parser.add_argument("--render", action='store_true', default=False, help="render results")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="batch size to use for evaluation")
    parser.add_argument("-f", "--force", action='store_true', default=False, help="Force evaluation")
    parser.add_argument("--snapshot-dir", default='..', help="path to log dir with snapshots to evaluate")

    args = parser.parse_args()
    batch_size = args.batch_size
    args = parse_config(args.config, args)
    args.batch_size = batch_size

    test_dataset_prefix = "test_dataset_"
    test_datasets = [arg for arg in dir(args) if arg.startswith(test_dataset_prefix)]

    alnum_only_prefix = "test_alnum_only_"
    alnum_only = [arg for arg in dir(args) if arg.startswith(alnum_only_prefix)]

    possible_snapshots = determine_snapshot_names(args.snapshot_dir)
    if len(args.iteration) > 0:
        assert int(args.iteration) in possible_snapshots['iterations'], "The iteration you want to analyze can not be found in the saved iterations"

    eval_results = defaultdict(list)

    for test_dataset_name, only_alnum in zip(test_datasets, alnum_only):
        dataset_path = getattr(args, test_dataset_name)
        test_dataset_name = test_dataset_name[len(test_dataset_prefix):]
        print(f"Testing {test_dataset_name}")

        dest_file_name = f"{test_dataset_name}_eval_results.json"

        command = "python"
        file = "evaluate.py"

        process_args = [
            "--gpu", str(args.gpu),
            dataset_path,
            args.snapshot_dir,
            f"{possible_snapshots['localizer']}_{args.iteration}",
            "--recognizer-name",
            f"{possible_snapshots['recognizer']}_{args.iteration}",
            "--char-map",
            args.char_map,
            "--results-path",
            dest_file_name,
            "--dataset-name",
            test_dataset_name,
        ]

        if getattr(args, only_alnum):
            print("stripping non alpha")
            process_args.append("--strip-non-alpha")

        if args.force:
            process_args.append("--force-reset")

        if args.render:
            process_args.extend([
                "--save-predictions",
                "--do-not-cut-bboxes",
                "--render-all-results",
                "-b",
                "1",
            ])
        else:
            process_args.extend([
                "-b",
                f"{args.batch_size}",
            ])

        subprocess.run([command, file] + process_args, check=True)

        with open(os.path.join(args.snapshot_dir, dest_file_name)) as f:
            eval_data = json.load(f)

        for snapshot_data in eval_data:
            snapshot_data['dataset'] = test_dataset_name
            eval_results[snapshot_data['snapshot_name']].append(snapshot_data)

    # now find best result
    best_result = find_best_result(eval_results)

    pprint.pprint(best_result)
    with open(os.path.join(args.snapshot_dir, "all_eval_results.json"), 'w') as f:
        json.dump(best_result, f, indent=4)
