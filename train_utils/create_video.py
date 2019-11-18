import argparse
import os
import re

from collections import namedtuple

import subprocess

import shutil
import tempfile

import sys

SUPPORTED_IMAGETYPES = [".png", ".jpg", ".jpeg"]
VIDEO_TYPES = [".mpeg", ".mp4"]
ImageData = namedtuple('ImageData', ['file_name', 'path'])


def get_filter(pattern):
    def filter_function(x):
        return int(re.search(pattern, x.file_name).group(1))
    return filter_function


def make_video(image_dir, dest_file, batch_size=1000, start=None, end=None, pattern=r"(\d+)"):
    dest_type = os.path.splitext(dest_file)[-1]
    if dest_type == ".gif":
        animation_creator = create_gif
        concatenate_function = concatenate_gifs
    elif dest_type in VIDEO_TYPES:
        animation_creator = create_video
        concatenate_function = concatenate_videos
    else:
        print("Output type {} is not supported".format(dest_type))
        sys.exit(1)

    sort_pattern = re.compile(pattern)

    image_files = os.listdir(image_dir)

    image_files = filter(lambda x: os.path.splitext(x)[-1] in SUPPORTED_IMAGETYPES, image_files)
    images = []

    print("loading images")
    for file_name in image_files:
        path = os.path.join(image_dir, file_name)
        images.append(ImageData(file_name=file_name, path=path))

    extract_number = get_filter(sort_pattern)
    if end is None:
        end = extract_number(max(images, key=extract_number))
    if start is None:
        start = 0

    print("sort and cut images")
    images_sorted = list(filter(
        lambda x: start <= extract_number(x) < end,
        sorted(images, key=extract_number)))

    print("creating temp file")
    temp_file = tempfile.NamedTemporaryFile(mode="w")
    video_dir = tempfile.mkdtemp()
    i = 1
    try:
        # create a bunch of videos and merge them later (saves memory)
        while i < len(images_sorted):
            image = images_sorted[i]
            if i % batch_size == 0:
                temp_file = animation_creator(i, temp_file, video_dir)
            else:
                print(image.path, file=temp_file)
            i += 1

        if i % batch_size != 0:
            print("creating last video")
            temp_file = animation_creator(i - 1, temp_file, video_dir)
        temp_file.close()

        # merge created videos
        concatenate_function(video_dir, dest_file)
    finally:
        shutil.rmtree(video_dir)


def create_video(i, temp_file, video_dir):
    process_args = [
        'convert',
        '-quality 100',
        '@{}'.format(temp_file.name),
        os.path.join(video_dir, "{}.mpeg".format(i))
    ]
    return run_animation_process(process_args, temp_file)


def create_gif(i, temp_file, gif_dir):
    process_args = [
        'convert',
        '-delay 10',
        '-loop 0',
        '@{}'.format(temp_file.name),
        os.path.join(gif_dir, "{}.gif".format(i))
    ]
    return run_animation_process(process_args, temp_file)


def concatenate_gifs(video_dir, dest_file):
    file_names = sorted(
        os.listdir(video_dir),
        key=lambda x: int(os.path.splitext(x.rsplit('/', 1)[-1])[0])
    )
    process_args = [
        'cat',
        ' '.join([os.path.join(video_dir, file_name) for file_name in file_names]),
        '> {}'.format(os.path.abspath(dest_file)),
    ]
    print(' '.join(process_args))
    subprocess.run(' '.join(process_args), shell=True, check=True, cwd=video_dir)


def concatenate_videos(video_dir, dest_file):
    process_args = [
        'ffmpeg',
        '-i concat:"{}"'.format(
            '|'.join(sorted(
                os.listdir(video_dir),
                key=lambda x: int(os.path.splitext(x.rsplit('/', 1)[-1])[0]))
            )
        ),
        '-c copy {}'.format(os.path.abspath(dest_file))
    ]
    print(' '.join(process_args))
    subprocess.run(' '.join(process_args), shell=True, check=True, cwd=video_dir)


def run_animation_process(process_args, temp_file):
    print(' '.join(process_args))
    temp_file.flush()
    subprocess.run(' '.join(process_args), shell=True, check=True)
    temp_file.close()
    temp_file = tempfile.NamedTemporaryFile(mode="w")
    return temp_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that creates a gif out of a number of given input images')
    parser.add_argument("image_dir", help="path to directory that contains all images that shall be converted to a gif/mpeg")
    parser.add_argument("dest_file", help="path to destination gif/mpeg file")
    parser.add_argument("--pattern", default=r"(\d+)", help="naming pattern to extract the ordering of the images")
    parser.add_argument("--batch-size", "-b", default=1000, type=int, help="batch size for processing, [default=1000]")
    parser.add_argument("-e", "--end", type=int, help="maximum number of images to put in gif")
    parser.add_argument("-s", "--start", type=int, help="frame to start")

    args = parser.parse_args()

    make_video(args.image_dir, args.dest_file, batch_size=args.batch_size, start=args.start, end=args.end, pattern=args.pattern)
