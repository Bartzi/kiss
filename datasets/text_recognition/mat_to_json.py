import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import argparse
import json

import os
import re

import scipy.io
from tqdm import tqdm

from iou.bbox import BBox, Vec2


def split_text(text):
    all_words = [word for instance in text for word in re.split(r'\s+', instance) if len(word) > 0]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tool that converts oxford gt mat file to json")
    parser.add_argument("mat_file", help="path to original oxford gt")
    parser.add_argument("-o", "--output", help="path to output file [default: 'mat_file'.json]")

    args = parser.parse_args()

    oxford_data = scipy.io.loadmat(args.mat_file)

    image_names = oxford_data['imnames'][0]
    word_bboxes = oxford_data['wordBB'][0]
    char_bboxes = oxford_data['charBB'][0]
    text = oxford_data['txt'][0]

    data = []
    for i, (image_name, bboxes, char_bbox, words) in tqdm(enumerate(zip(image_names, word_bboxes, char_bboxes, text)), total=len(image_names)):
        words = split_text(words)
        image_name = image_name[0]
        bboxes, aabbs = get_bboxes(bboxes)
        char_boxes, char_aabbs = get_bboxes(char_bbox)

        assert len(words) == len(bboxes), f"Number of words and bboxes does not match for image: {image_name}, id: {i}"

        data.append({
            "file_name": image_name,
            "text": words,
            "oriented_bounding_boxes": bboxes,
            "bounding_boxes": aabbs,
            "oriented_char_bounding_boxes": char_boxes,
            "char_bounding_boxes": char_aabbs,
        })

    dest_file_name = f"{os.path.splitext(args.mat_file)[0]}.json"
    with open(dest_file_name, 'w') as f:
        json.dump(data, f, indent=4)
