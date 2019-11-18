import argparse
import json

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import re
import scipy.io
from PIL import Image
from tqdm import tqdm

from iou.bbox import BBox, Vec2, AxisAlignedBBox


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


def get_relative_box_position(anchor, box):
    anchor_vec = Vec2(anchor.left, anchor.top)
    return BBox(
        box.top_left - anchor_vec,
        box.top_right - anchor_vec,
        box.bottom_right - anchor_vec,
        box.bottom_left - anchor_vec,
    )


def main(args):
    gt_data = scipy.io.loadmat(args.gt_file)

    image_names = gt_data['imnames'][0]
    word_bboxes = gt_data['wordBB'][0]
    char_bboxes = gt_data['charBB'][0]
    text = gt_data['txt'][0]

    source_image_dir = os.path.dirname(args.gt_file)

    iterator = tqdm(zip(image_names, word_bboxes, char_bboxes, text), total=len(image_names))

    generated_gt = []
    length_of_longest_word = 0
    for image_name, word_boxes, char_boxes, words in iterator:
        words = split_text(words)
        image_name = image_name[0]
        bboxes, aabbs = get_bboxes(word_boxes)
        char_boxes, char_aabbs = get_bboxes(char_boxes)

        try:
            with Image.open(os.path.join(source_image_dir, image_name)) as the_image:
                char_shift = 0
                for i in range(len(aabbs)):
                    crop = aabbs[i].crop_from_image(the_image)
                    destination_dir = os.path.join(args.destination, os.path.dirname(image_name))
                    os.makedirs(destination_dir, exist_ok=True)
                    destination_file_name = f"{os.path.splitext(image_name)[0]}_{i}.png"

                    crop.save(os.path.join(args.destination, destination_file_name))
                    boxes = char_boxes[char_shift:char_shift+len(words[i])]
                    boxes = [get_relative_box_position(aabbs[i], char_box) for char_box in boxes]
                    generated_gt.append({
                        "file_name": destination_file_name,
                        "text": [[char] for char in words[i]],
                        "bounding_boxes": boxes,
                    })
                    char_shift += len(words[i])

                    if len(words[i]) > length_of_longest_word:
                        length_of_longest_word = len(words[i])
        except Exception as e:
            print(e)
            print(image_name)

    generated_gt.insert(0, {"num_chars": 1, "num_words": length_of_longest_word})

    with open(os.path.join(args.destination, "gt.json"), "w") as f:
        json.dump(generated_gt, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that takes oxford detection gt and crops all words using their aabb")
    parser.add_argument("gt_file", help="path to oxford gt file")
    parser.add_argument("destination", help="path to destination dir, where you want to save the cropped images")

    args = parser.parse_args()
    main(args)
