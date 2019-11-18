import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import argparse
import json

import os
import re

import h5py
from tqdm import tqdm

from iou.bbox import BBox, Vec2


def split_text(text):
    all_words = [word for instance in text for word in re.split(r'\s+', instance.decode('utf-8')) if len(word) > 0]
    return all_words


def get_bboxes(bboxes):
    if len(bboxes.shape) == 2:
        bboxes = bboxes[..., None]

    bboxes = bboxes.transpose(2, 1, 0)
    boxes = []
    aabbs = []
    for box in bboxes:
        box = BBox._make([Vec2._make(b).to_int() for b in box])
        boxes.append(box)

        aabbs.append(box.to_aabb())

    return boxes, aabbs


def build_file_name(name):
    name, ext = os.path.splitext(name)
    ext = ext.split('_')

    if len(ext) > 1:
        name = name + '_'.join(ext[1:])

    new_name = f"{name}.png"
    return new_name


def save_image(image_name, image_data):
    image = Image.fromarray(image_data)
    image_name = build_file_name(image_name)
    image.save(image_name)
    return image_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tool that converts oxford gt mat file to json")
    parser.add_argument("mat_file", help="path to original oxford gt")
    parser.add_argument("output_dir", help="path to output dir")

    args = parser.parse_args()

    oxford_data = h5py.File(args.mat_file)['data']

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    data = []
    for image_name in tqdm(oxford_data, total=len(oxford_data)):
        dataset = oxford_data[image_name]
        image_path = os.path.join(image_dir, image_name)

        image_path = save_image(image_path, dataset[:])
        # image_path = build_file_name(image_path)
        words = split_text(dataset.attrs['txt'])
        bboxes, aabbs = get_bboxes(dataset.attrs['wordBB'])

        assert len(words) == len(bboxes), f"Number of words and bboxes does not match for image: {image_name}"

        data.append({
            "file_name": os.path.relpath(image_path, args.output_dir),
            "text": words,
            "oriented_bounding_boxes": bboxes,
            "bounding_boxes": aabbs,
        })

    dest_file_name = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.mat_file))[0]}.json")
    print(f"saving to {dest_file_name}")

    with open(dest_file_name, 'w') as f:
        json.dump(data, f, indent=4)
