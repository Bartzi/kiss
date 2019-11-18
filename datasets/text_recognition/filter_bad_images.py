import argparse
import json

import chainer
import os

import numpy as np

from PIL import Image
from tqdm import tqdm


def test_image(base_dir, file_name):
    try:
        with Image.open(os.path.join(base_dir, file_name)) as the_image:
            the_image.convert('L').convert("RGB")
            the_image = the_image.resize((200, 64), Image.LANCZOS)
            image = np.array(the_image, chainer.get_dtype())
            assert image.max() > 0
            return True
    except Exception:
        return False


def main(args):
    with open(args.gt_file) as f:
        gt_data = json.load(f)

    good_gt_data = []
    base_dir = os.path.dirname(args.gt_file)
    for image_info in tqdm(gt_data):
        if 'file_name' not in image_info:
            good_gt_data.append(image_info)
            continue

        if test_image(base_dir, image_info['file_name']):
            good_gt_data.append(image_info)

    print(f"original num images: {len(gt_data)}, new num images: {len(good_gt_data)}, delta: {len(gt_data) - len(good_gt_data)}")

    gt_file_name, extension = os.path.splitext(args.gt_file)
    with open(os.path.join(base_dir, f"{gt_file_name}_filtered{extension}"), 'w') as f:
        json.dump(good_gt_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that loads each image of a given dataset and keeps only those that can be loaded and seem to be okay")
    parser.add_argument("gt_file", help="Path to gt file in json format")

    args = parser.parse_args()
    main(args)
