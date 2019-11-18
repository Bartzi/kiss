import argparse
import json
import os

import random


def main(args):
    with open(args.gt_file) as f:
        gt_data = json.load(f)

    metadata = None
    if "file_name" not in gt_data[0]:
        # we do not want to distribute the metadata, but we have to save it for later
        metadata = gt_data.pop(0)

    number_of_images = len(gt_data)
    num_validation_images = int(number_of_images * args.val_ratio)
    num_reference_images = int(number_of_images * args.ref_ratio)

    random.shuffle(gt_data)

    validation_images = gt_data[:num_validation_images]
    reference_images = gt_data[num_validation_images:num_validation_images + num_reference_images]
    train_images = gt_data[num_validation_images + num_reference_images:]

    gt_dir = os.path.dirname(args.gt_file)

    with open(os.path.join(gt_dir, "reference_base.json"), "w") as f:
        json.dump(reference_images, f, indent=2)

    with open(os.path.join(gt_dir, "validation.json"), "w") as f:
        if metadata is not None:
            validation_images.insert(0, metadata)
        json.dump(validation_images, f, indent=2)

    with open(os.path.join(gt_dir, "train.json"), "w") as f:
        if metadata is not None:
            train_images.insert(0, metadata)
        json.dump(train_images, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that takes a gt json file and creates a training, validation and reference gt")
    parser.add_argument("gt_file")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="ratio for validation images")
    parser.add_argument("--ref-ratio", type=float, default=0.1, help="ratio of reference images")

    args = parser.parse_args()
    main(args)
