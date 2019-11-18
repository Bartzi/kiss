import argparse
import json
from collections import defaultdict

import os

import numpy as np
import re

from tqdm import tqdm


METADATA_KEYS = ['num_chars', 'num_words']
ITER_KEY = 'text'

non_alpha_numeric_pattern = re.compile(r"[^0-9a-zA-Z]")


def filter_npz_file(gt_file):
    filtered_gt = defaultdict(list)

    with np.load(gt_file, allow_pickle=True) as handle:
        assert int(handle['num_chars'][0]) == 1, "This filtering is intended for datasets with already cropped words"

        filtered_gt['num_chars'] = handle['num_chars'][0]
        filtered_gt['num_words'] = handle['num_words'][0]

        keys_to_copy = [key for key in handle.keys() if key not in METADATA_KEYS and key != ITER_KEY]

        main_data = handle[ITER_KEY]
        key_data = {key: handle[key] for key in keys_to_copy}

        target_shape = main_data.shape
        keys_to_copy = [key for key in keys_to_copy if key_data[key].shape == target_shape]

        dataset_length = len(main_data)
        for index in tqdm(range(dataset_length), total=dataset_length):
            if re.search(non_alpha_numeric_pattern, main_data[index]):
                continue
            filtered_gt[ITER_KEY].append(main_data[index])
            for key in keys_to_copy:
                filtered_gt[key].append(key_data[key][index])

    print(f"saving the following keys: {list(filtered_gt.keys())}")
    dest_file_name = f"{os.path.splitext(gt_file)[0]}_alpha_numeric.npz"
    with open(dest_file_name, 'wb') as f:
        np.savez_compressed(f, **filtered_gt)


def main(args):
    for gt_file in tqdm(args.gt_files):
        filter_npz_file(gt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that filters given gt file by keeping only images with words of a max specific length")
    parser.add_argument("gt_files", nargs="+", help="path to gt files that shall be filtered")

    args = parser.parse_args()
    main(args)
