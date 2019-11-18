import argparse
import json

import os
from pathlib import Path

import csv


def main(args):
    original_gt_path = Path(args.icdar_gt)
    image_path = Path(args.icdar_image_dir)

    relative_path = image_path.relative_to(original_gt_path.parent)

    with open(original_gt_path, newline='') as f:
        reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL)
        gt_data = [l for l in reader]

    converted_gt_data = []
    for data in gt_data:
        file_name = data[0]
        word = data[1]
        if len(data) > 2:
            word = ','.join(data[1:])
        word = word.strip().strip('"')

        file_name = relative_path / file_name
        converted_gt_data.append({
            "file_name": str(file_name),
            "text": [[char] for char in word],
            "bounding_boxes": [[0, 0, 0, 0] for _ in word],
        })

    destination_dir = os.path.dirname(original_gt_path)
    with open(os.path.join(destination_dir, "icdar_gt.json"), 'w') as f:
        json.dump(converted_gt_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ICDAR 2013 Focused Scene Text GT to our format")
    parser.add_argument("icdar_gt", help='path to original icdar gt')
    parser.add_argument("icdar_image_dir", help='path to dir containing images')

    args = parser.parse_args()
    main(args)
