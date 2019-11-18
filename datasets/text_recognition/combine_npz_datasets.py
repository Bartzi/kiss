import argparse
import json
import os
from collections import defaultdict

import numpy as np

from tqdm import tqdm


def main(args):
    combined_gt = defaultdict(list)

    for dataset in tqdm(args.datasets):
        with np.load(dataset, allow_pickle=True) as f:
            for key, item in tqdm(f.items()):
                if key != 'file_name':
                    combined_gt[key].extend(item)
                    continue

                # adapt paths of images
                adapted_paths = []
                source_root = os.path.realpath(os.path.dirname(dataset))
                destination_root = os.path.realpath(os.path.dirname(args.destination))

                for image_path in item:
                    image_path = os.path.join(source_root, image_path)
                    same_prefix = os.path.commonpath([destination_root, image_path])
                    new_path = image_path[len(same_prefix) + 1:]
                    adapted_paths.append(new_path)

                combined_gt[key].extend(adapted_paths)

    with open(args.destination, 'wb') as f:
        np.savez_compressed(f, **combined_gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple datasets in our json format in one file")
    parser.add_argument("datasets", nargs='+', help='Path to all npz files you want to combine')
    parser.add_argument("--destination", required=True, help='path where new file shall be saved')

    args = parser.parse_args()
    main(args)
