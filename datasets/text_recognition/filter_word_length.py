import argparse
import json
from collections import defaultdict

import os

import numpy as np

from tqdm import tqdm


METADATA_KEYS = ['num_chars', 'num_words']
ITER_KEY = 'text'


def filter_json_file(gt_file, max_word_length):
    with open(gt_file) as f:
        gt_data = json.load(f)

    metadata = gt_data.pop(0)
    assert "bounding_boxes" not in metadata, "There is not metadata in this gt file? Is this correct?"
    assert metadata["num_chars"] == 1, "This filtering is intended for datasets with already cropped words"

    metadata["num_words"] = max_word_length

    filtered_gt = [image_data for image_data in tqdm(gt_data) if len(image_data['text']) <= max_word_length]
    filtered_gt.insert(0, metadata)

    dest_file_name = f"{os.path.splitext(gt_file)[0]}_filtered_{max_word_length}.json"
    with open(dest_file_name, "w") as f:
        json.dump(filtered_gt, f, indent=2)


def filter_npz_file(gt_file, max_word_length):
    filtered_gt = defaultdict(list)

    with np.load(gt_file, allow_pickle=True) as handle:
        assert int(handle['num_chars'][0]) == 1, "This filtering is intended for datasets with already cropped words"

        filtered_gt['num_chars'] = handle['num_chars'][0]
        filtered_gt['num_words'] = np.array(max_word_length)

        keys_to_copy = [key for key in handle.keys() if key not in METADATA_KEYS and key != ITER_KEY]

        main_data = handle[ITER_KEY]
        key_data = {key: handle[key] for key in keys_to_copy}

        target_shape = main_data.shape
        keys_to_copy = [key for key in keys_to_copy if key_data[key].shape == target_shape]

        dataset_length = len(main_data)
        for index in tqdm(range(dataset_length), total=dataset_length):
            if len(main_data[index]) > max_word_length:
                continue
            filtered_gt[ITER_KEY].append(main_data[index])
            for key in keys_to_copy:
                filtered_gt[key].append(key_data[key][index])

    print(f"saving the following keys: {list(filtered_gt.keys())}")
    dest_file_name = f"{os.path.splitext(gt_file)[0]}_filtered_{max_word_length}.npz"
    with open(dest_file_name, 'wb') as f:
        np.savez_compressed(f, **filtered_gt)


def main(args):
    for gt_file in tqdm(args.gt_files):
        if args.npz:
            func = filter_npz_file
        else:
            func = filter_json_file

        func(gt_file, args.max_word_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that filters given gt file by keeping only images with words of a max specific length")
    parser.add_argument("max_word_length", type=int, help="max length of a word in an image")
    parser.add_argument("gt_files", nargs="+", help="path to gt files that shall be filtered")
    parser.add_argument("--npz", action='store_true', default=False, help="indicate that gt files are saved as npz not json")

    args = parser.parse_args()
    main(args)
