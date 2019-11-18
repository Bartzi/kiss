import argparse
import os
from PIL import Image
from tqdm import tqdm

from common.datasets.image_dataset import BaseNPZImageDataset


def main(args):
    dataset = BaseNPZImageDataset((64, 200), args.npz_file, root=os.path.dirname(args.npz_file), transform_probability=1.0)

    for i in tqdm(range(args.num_iterations), total=args.num_iterations):
        image = dataset.get_example(1)
        image = image.transpose(1, 2, 0)
        image *= 255
        image = image.astype('uint8')
        image = Image.fromarray(image)
        dest_file_name = os.path.join(args.dest, f"{i}.png")
        image.save(dest_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that takes an image and uses shows what imgaug does")
    parser.add_argument("npz_file", help="path to npz that holds a dataset to test augmentations with")
    parser.add_argument("dest", help="path where to save example images")
    parser.add_argument("-n", "--num-iterations", type=int, default=100, help="the number of images to save")

    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    main(args)
