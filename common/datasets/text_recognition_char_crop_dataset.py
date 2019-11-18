import chainer
import json
import numpy as np
from chainer.backends import cuda
from chainer.functions import spatial_transformer_grid, spatial_transformer_sampler

from common.datasets.image_dataset import BaseImageDataset
from common.datasets.text_recognition_image_dataset import TextRecognitionImageDataset
from iou.bbox import Vec2, BBox, AxisAlignedBBox
from train_utils.dataset_utils import retry_get_example
from train_utils.datatypes import Size


class TextRecognitionImageCharCropDataset(TextRecognitionImageDataset):

    def __init__(self, *args, **kwargs):
        self.reverse_transform_params = kwargs.pop('reverse', False)
        self.device_id = kwargs.pop('gpu_id', -1)
        kwargs['resize_after_load'] = False
        super().__init__(*args, **kwargs)

        self.xp = np if self.device_id < 0 else cuda.cupy

    def transform_params(self, i):
        return self.get_gt_item('transform_params', i)

    def apply_transform_params(self, image, transform_params):
        image = self.xp.tile(image[np.newaxis, ...], (len(transform_params), 1, 1, 1))

        transform_grid = spatial_transformer_grid(transform_params, self.image_size)
        cropped_image = spatial_transformer_sampler(image, transform_grid).array
        return cropped_image

    def crop_chars(self, image, i):
        transform_params = self.transform_params(i)

        with cuda.get_device_from_id(self.device_id):
            if self.reverse_transform_params:
                transform_params = transform_params[::-1]
            transform_params = self.xp.array(transform_params, dtype=chainer.get_dtype())
            transform_params = transform_params.reshape(-1, 2, 3)

            return self.apply_transform_params(image, transform_params)

    @retry_get_example
    def get_example(self, i):
        image_data = super().get_example(i)
        image_data['image'] = self.crop_chars(image_data['image'], i)
        return image_data


class TextRecognitionEvenCharCropDataset(TextRecognitionImageCharCropDataset):

    def __init__(self, *args, **kwargs):
        self.reverse_transform_params = kwargs.pop('reverse', False)
        self.device_id = kwargs.pop('gpu_id', -1)
        kwargs['resize_after_load'] = False
        super(TextRecognitionImageCharCropDataset, self).__init__(*args, **kwargs)

        self.xp = np if self.device_id < 0 else cuda.cupy

    def crop_chars(self, image, i):
        image_size = Size(*image.shape[-2:])
        image_bbox = BBox.from_axis_aligned_bbox(AxisAlignedBBox(0, 0, image_size.width, image_size.height))

        transform_params = []
        crop_vector = Vec2(x=image_bbox.top_right.x - image_bbox.top_left.x,
                           y=image_bbox.top_right.y - image_bbox.top_left.y)
        start_points = [(crop_vector * (i / self.num_words_per_image)).to_int() for i in
                        range(self.num_words_per_image)]
        crop_bbox_width = crop_vector * (1 / (self.num_words_per_image / 4))  # divide by four for 75% overlap
        box_overlap = Vec2((start_points[-1].x + crop_bbox_width.x - image_size.width) // 2, 0)

        start_points = [point - box_overlap for point in start_points]

        with cuda.get_device_from_id(self.device_id):
            for start_point in start_points:
                top_left = image_bbox.top_left + start_point
                bottom_left = image_bbox.bottom_left + start_point
                top_right = image_bbox.top_left + start_point + crop_bbox_width
                bottom_right = image_bbox.bottom_left + start_point + crop_bbox_width
                crop_bbox = BBox(top_left, top_right, bottom_right, bottom_left)

                transform_params.append(
                    crop_bbox.get_affine_transform_params(Vec2(x=image_size.width, y=image_size.height), xp=self.xp)
                )

            transform_params = self.xp.stack(transform_params, axis=0)
            return self.apply_transform_params(image, transform_params)
