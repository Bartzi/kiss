import argparse

import csv

import json
import os
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that converts Jaderberg SynthText GT to our format")
    parser.add_argument("gt_file")
    parser.add_argument("output_file")

    args = parser.parse_args()

    converted_data = []
    max_length = 0
    chars = set()
    with open(args.gt_file) as handle:
        reader = csv.reader(handle, delimiter=' ')
        for file_name, word_id in tqdm(reader):
            word = file_name.split('_')[-2]
            chars = chars.union({char for char in word})
            if len(word) > max_length:
                max_length = len(word)

            converted_data.append({
                "file_name": file_name,
                "text": [[char] for char in word],
            })

    converted_data.insert(0, {
        "num_chars": max_length,
        "num_words": 1,
    })

    chars = list(chars)
    chars.append(chr(9250))

    char_map = {i: ord(c) for i, c in enumerate(chars)}
    char_map["metadata"] = {"blank_label": len(chars) - 1}

    output_dir = os.path.dirname(args.output_file)

    with open(args.output_file, 'w') as handle:
        json.dump(converted_data, handle, indent=4)

    with open(os.path.join(output_dir, "char_map.json"), "w") as handle:
        json.dump(char_map, handle, indent=4)
