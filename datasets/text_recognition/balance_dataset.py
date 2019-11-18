import argparse
import json
import os
import random

from collections import defaultdict
from pprint import pprint

import numpy as np

from tqdm import tqdm

from filter_word_length import METADATA_KEYS


def balance_json_file(args):
    with open(args.gt_file) as f:
        gt_data = json.load(f)

    text_lengths = defaultdict(list)

    metadata = gt_data.pop(0)
    assert 'text' not in metadata, "There is no metadata in the file?"

    for image_info in tqdm(gt_data):
        if 'text' not in image_info:
            continue

        text_lengths[len(image_info['text'])].append(image_info)

    distributions = {k: len(v) for k, v in text_lengths.items()}
    pprint(distributions)

    if args.max_num_samples is None or args.dry_run:
        return

    text_lengths = {k: random.sample(v, min(args.max_num_samples, len(v))) for k, v in text_lengths.items()}
    pprint({k: len(v) for k, v in text_lengths.items()})

    balanced_gt = [item for items in text_lengths.values() for item in items]
    balanced_gt.insert(0, metadata)

    with open(os.path.join(os.path.dirname(args.gt_file), args.destination_name), 'w') as f:
        json.dump(balanced_gt, f, indent=2)


def balance_npz_file(args):
    balanced_gt = {}
    with np.load(args.gt_file, allow_pickle=True) as handle:
        assert int(handle['num_chars']) == 1, "This balancing is intended for datasets with already cropped words"

        keys_to_balance = [key for key in handle.keys() if key not in METADATA_KEYS]
        text_lengths = defaultdict(list)

        for i, word in enumerate(tqdm(handle['text'])):
            text_lengths[len(word)].append(i)

        distributions = {k: len(v) for k, v in text_lengths.items()}
        pprint(distributions)

        if args.max_num_samples is None or args.dry_run:
            return

        text_lengths = {k: random.sample(v, min(args.max_num_samples, len(v))) for k, v in text_lengths.items()}
        pprint({k: len(v) for k, v in text_lengths.items()})

        indices_to_keep = [index for length in text_lengths.values() for index in length]
        print(len(indices_to_keep))

        for key in keys_to_balance:
            balanced_gt[key] = handle[key][indices_to_keep]

        balanced_gt['num_chars'] = handle['num_chars']
        balanced_gt['num_words'] = handle['num_words']

    assert all(balanced_gt[key].shape[0] == balanced_gt['text'].shape[0] for key in keys_to_balance), "shapes are not equal {}".format({k: v.shape for k, v in balanced_gt.items()})

    with open(args.destination_name, 'wb') as f:
        np.savez_compressed(f, **balanced_gt)


def main(args):
    if os.path.splitext(args.gt_file)[-1] == ".npz":
        balance_npz_file(args)
    else:
        balance_json_file(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that tries to balance the dataset in order to avoid predictions of the majority class")
    parser.add_argument("gt_file", help="path to gt that is to be balanced")
    parser.add_argument("destination_name", help="name of the file that is to be generated (will be placed in same dir as original gt file)")
    parser.add_argument("-m", "--max-num-samples", type=int, default=None, help="maximum number of samples to keep for each text length (if not given, only show statistics)")
    parser.add_argument("--dry-run", action='store_true', default=False, help="only print statistics but do not balance dataset")

    args = parser.parse_args()
    main(args)
