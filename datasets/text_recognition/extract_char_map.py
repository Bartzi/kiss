import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm


def main(args):
    with open(args.gt_file) as f:
        gt_data = json.load(f)

    chars = defaultdict(int)
    for image_data in tqdm(gt_data):
        if 'text' not in image_data:
            continue
        words = image_data['text']
        for word in words:
            for char in word:
                chars[char] += 1

    print(f"Num Chars: {len(chars)}")

    char_map = {i: c for i, c in enumerate(sorted(chars.keys(), key=ord))}
    char_map[len(chars)] = args.blank_label
    char_map['metadata'] = {"blank_label": len(chars)}
    if args.add_bos_token:
        bos_token_id = len(chars) + 1
        char_map['metadata']['bos_token'] = bos_token_id
        char_map[bos_token_id] = args.bos_token

    with open(os.path.join(os.path.dirname(args.gt_file), f"{args.name}.json"), "w") as f:
        json.dump(char_map, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="tool that creates a char map from a given gt file (saves char map in the same directory as gt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("gt_file", help="Path to gt file")
    parser.add_argument("-n", "--name", default="char_map", help="name of the char map file")
    parser.add_argument("-b", "--blank-label", default=chr(9250), help="blank char")
    parser.add_argument("--bos-token", default=chr(152), help='begin of sequence token for transformer')
    parser.add_argument("--add-bos-token", action='store_true', default=False, help="whether to add a BOS token to the char map")

    args = parser.parse_args()
    main(args)
