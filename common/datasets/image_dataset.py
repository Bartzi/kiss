import json

import chainer
import numpy
numpy.random.bit_generator = numpy.random._bit_generator
import os
import six
from PIL import Image
from chainer.dataset import DatasetMixin
from chainer.datasets.image_dataset import _check_pillow_availability

from train_utils.dataset_utils import retry_get_example
from train_utils.datatypes import Size


import imgaug
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def aspect_ratio_preserving_resize(the_image, image_size):
    ratio = min(image_size.width / the_image.width, image_size.height / the_image.height)
    the_image = the_image.resize((int(ratio * the_image.width), int(ratio * the_image.height)), Image.LANCZOS)

    # paste resized image into blank image to have image of correct input size
    blank_image = Image.new(the_image.mode, (image_size.width, image_size.height))
    # determine center of image and paste new image to be in the center!
    paste_start_x = image_size.width // 2 - the_image.width // 2
    paste_start_y = image_size.height // 2 - the_image.height // 2
    blank_image.paste(the_image, (paste_start_x, paste_start_y))

    return blank_image


def prepare_image(the_image, image_size, xp, keep_aspect_ratio, normalize=False, do_resize=True):
    if do_resize:
        if keep_aspect_ratio:
            the_image = aspect_ratio_preserving_resize(the_image, image_size)
        else:
            the_image = the_image.resize((image_size.width, image_size.height), Image.LANCZOS)

    image = xp.array(the_image, chainer.get_dtype())

    if normalize:
        image -= image.min()
        image /= max(image.max(), 1)
    else:
        image /= 255
    # image = image * 2 - 1
    return xp.transpose(image, (2, 0, 1))


class BaseImageDataset(DatasetMixin):

    def __init__(self, paths, image_size, root='.', dtype=None, transform_probability=0, use_imgaug=True, keep_aspect_ratio=False, image_mode='RGB', full_normalize=False, resize_after_load=True):
        _check_pillow_availability()
        assert isinstance(paths, six.string_types), "paths must be a file name!"
        assert os.path.splitext(paths)[-1] == ".json", "You have to supply gt information as json file!"

        if not isinstance(image_size, Size):
            image_size = Size(*image_size)

        with open(paths) as handle:
            self.gt_data = json.load(handle)

        self.root = root
        self.dtype = chainer.get_dtype(dtype)
        self.image_size = image_size
        self.transform_probability = transform_probability
        self.use_imgaug = use_imgaug
        self.keep_aspect_ratio = keep_aspect_ratio
        self.image_mode = image_mode
        self.augmentations = self.init_augmentations()
        self.full_normalize = full_normalize  # normalize each image to be in range of [0, 1] even if brightest pixel is != 255
        self.resize_after_load = resize_after_load  # resize the image to self.image_size after loading

    def shrink_dataset(self, num_samples):
        self.gt_data = self.gt_data[:num_samples]

    def init_augmentations(self):
        if self.transform_probability > 0 and self.use_imgaug:
            augmentations = iaa.Sometimes(
                self.transform_probability,
                iaa.Sequential([
                    iaa.SomeOf(
                        (1, None),
                        [
                            iaa.AddToHueAndSaturation(iap.Uniform(-20, 20), per_channel=True),
                            iaa.GaussianBlur(sigma=(0, 1.0)),
                            iaa.LinearContrast((0.75, 1.0)),
                            iaa.PiecewiseAffine(scale=(0.01, 0.02), mode='edge'),
                        ],
                        random_order=True
                    ),
                    iaa.Resize(
                        {"height": (16, self.image_size.height), "width": "keep-aspect-ratio"},
                        interpolation=imgaug.ALL
                    ),
                ])
            )
        else:
            augmentations = None
        return augmentations

    def maybe_augment(self, image):
        if self.augmentations is not None:
            image_data = numpy.asarray(image)
            image_data = self.augmentations.augment_image(image_data)
            image = Image.fromarray(image_data)

        return image

    def __len__(self):
        return len(self.gt_data)

    def get_example(self, i):
        gt_data = self.gt_data[i]
        image = self.load_image(gt_data['file_name'])
        return image

    def load_image(self, file_name):
        with Image.open(os.path.join(self.root, file_name)) as the_image:
            the_image = the_image.convert(self.image_mode).convert("RGB")
            the_image = self.maybe_augment(the_image)
            image = prepare_image(
                the_image,
                self.image_size,
                numpy,
                self.keep_aspect_ratio,
                normalize=self.full_normalize,
                do_resize=self.resize_after_load,
            )
        return image


class BaseNPZImageDataset(BaseImageDataset):

    def __init__(self, image_size, npz_file=None, memory_manager=None, base_name=None, root='.', dtype=None, transform_probability=0, use_imgaug=True, keep_aspect_ratio=False, image_mode='RGB', full_normalize=False, resize_after_load=True):
        _check_pillow_availability()

        if not isinstance(image_size, Size):
            image_size = Size(*image_size)

        self.shared_buffers = []
        self.root = root
        self.dtype = chainer.get_dtype(dtype)
        self.image_size = image_size
        self.transform_probability = transform_probability
        self.use_imgaug = use_imgaug
        self.keep_aspect_ratio = keep_aspect_ratio
        self.image_mode = image_mode
        self.full_normalize = full_normalize  # normalize each image to be in range of [0, 1] even if brightest pixel is != 255
        self.resize_after_load = resize_after_load  # resize the image to self.image_size after loading

        if npz_file is not None:
            assert isinstance(npz_file, six.string_types), "paths must be a file name!"
            assert os.path.splitext(npz_file)[-1] == ".npz", "You have to supply gt information as npz file!"

            with numpy.load(npz_file, allow_pickle=True) as gt_data:
                self.gt_data = self.copy_npz_data_to_ram(gt_data)
            self.memory_manager = None
            self.base_name = None
            self.length = len(self.gt_data['file_name'])
        else:
            assert memory_manager is not None, "If you do not specify an npz file, you must specify a memory manager!"
            assert base_name is not None, "If you want to use shared memory, you'll need to supply a base name for each dataset"
            self.gt_data = None
            self.memory_manager = memory_manager
            self.base_name = base_name
            self.length = self.memory_manager.get_shape(self.base_name, 'file_name').pop(0)

        self.augmentations = self.init_augmentations()

    def file_names(self, i):
        return str(self.get_gt_item('file_name', i))

    def get_gt_item(self, key, index):
        if self.gt_data is not None:
            return self.gt_data[key][index]

        return self.get_shared_gt_data(key, index)

    def get_shared_gt_data(self, file_name, index):
        return self.memory_manager.get_data_member(self.base_name, file_name, index).view()

    def copy_npz_data_to_ram(self, gt_data):
        copied_gt_data = {}
        datasets = {key: gt_data[key] for key in gt_data.keys()}
        for key, data in datasets.items():
            array = numpy.ndarray(data.shape, dtype=data.dtype)

            if len(array.shape) == 0:
                # a single value array, we need to have a shape of at least 1
                array = array.reshape(1)

            # we have to copy the data from the file otherwise we won't the data once the file is closed
            array[:] = data
            copied_gt_data[key] = array
        return copied_gt_data

    def get_example(self, i):
        file_name = self.file_names(i)
        image = self.load_image(file_name)

        return image

    def __len__(self):
        return self.length


class ImageDataset(BaseImageDataset):

    @retry_get_example
    def get_example(self, i):
        return super().get_example(i)
