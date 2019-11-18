import argparse
import json
import os

from collections import defaultdict

import numpy as np

from tqdm import tqdm


def can_item_be_copied(text, inverse_char_map, ):
    chars_to_keep = [char for char in text if char[0] in inverse_char_map]
    if len(chars_to_keep) != len(text):
        return False
    return True


def main(args):
    with open(args.json_file) as f:
        gt_data = json.load(f)

    with open(args.char_map) as f:
        char_map = json.load(f)
        del char_map['metadata']
        inverse_char_map = {v: k for k, v in char_map.items()}

    converted_gt = defaultdict(list)

    discard_images = 0

    for item in tqdm(gt_data):
        if item is None:
            continue

        if 'text' in item.keys() and not can_item_be_copied(item['text'], inverse_char_map):
            discard_images += 1
            continue

        for key, value in item.items():
            if key == 'bounding_boxes':
                # we do not need any bounding boxes!
                continue
            if isinstance(value, list):
                if np.array(value).dtype.kind not in set('buifc'):
                    value = ''.join([v[0] for v in value])
                value = np.array(value)
            converted_gt[key].append(value)

    converted_gt = dict(converted_gt)
    for key in converted_gt.keys():
        converted_gt[key] = np.array(converted_gt[key])

    if args.destination is None:
        file_path = os.path.splitext(args.json_file)[0]
        args.destination = f"{file_path}.npz"

    with open(args.destination, 'wb') as f:
        np.savez_compressed(f, **converted_gt)

    print(f"done, had to discard {discard_images} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that converts text recognition json into npz format, hopefully helping to conserve runtime memory")
    parser.add_argument("json_file", help="path to json to convert")
    parser.add_argument("char_map", help="path to char map for filtering bad chars")
    parser.add_argument("--destination", help="path where resulting npz shall be saved (default is same name/location as json but as npz)")

    args = parser.parse_args()
    main(args)
