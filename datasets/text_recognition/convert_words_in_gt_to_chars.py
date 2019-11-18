import argparse
import json

from tqdm import tqdm


def main(args):
    for gt_file in tqdm(args.gt_files):
        with open(gt_file) as f:
            gt_data = json.load(f)

        for image_data in tqdm(gt_data):
            if 'text' not in image_data:
                continue

            word = image_data['text'][0]
            image_data['text'] = [[char] for char in word]

        with open(gt_file, "w") as f:
            json.dump(gt_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that takes previously generated gt that is supposed to be char based, but is actually word based and converts the text part to individual chars")
    parser.add_argument("gt_files", nargs="+", help="path to gt file to be transformed")

    args = parser.parse_args()
    main(args)

