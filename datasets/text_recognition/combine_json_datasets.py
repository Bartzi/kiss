import argparse
import json

from tqdm import tqdm


def main(args):
    combined_gt = []

    for dataset in tqdm(args.datasets):
        with open(dataset) as f:
            gt_data = json.load(f)

            metadata = gt_data.pop(0)
            if len(combined_gt) == 0:
                combined_gt.append(metadata)

            combined_gt.extend(gt_data)

    with open(args.destination, 'w') as f:
        json.dump(combined_gt, f, indent="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple datasets in our json format in one file")
    parser.add_argument("datasets", nargs='+', help='Path to all json files you want to combine')
    parser.add_argument("--destination", required=True, help='path where new file shall be saved')

    args = parser.parse_args()
    main(args)
