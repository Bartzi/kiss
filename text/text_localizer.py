from functools import reduce

import chainer
import chainer.functions as F
from chainer import Chain
from chainer.backends import cuda

from insights.visual_backprop import VisualBackprop
from resnet.resnet_gn import ResNet
from train_utils.autocopy import maybe_copy
from train_utils.datatypes import Size


class TextLocalizer(Chain):

    def __init__(self, out_size, **kwargs):
        self.transform_rois_to_grayscale = kwargs.pop('transform_rois_to_grayscale', False)
        self.num_bboxes_to_localize = kwargs.pop('num_bboxes_to_localize', 1)
        self.dropout_ratio = kwargs.pop('dropout_ratio', 0)
        self.box_offset_side_length = kwargs.pop('box_offset_side_length', 3)
        self.box_offset_factor = kwargs.pop('box_offset_factor', 20)
        self.features_per_timestep = kwargs.pop('features_per_timestep', 256)

        super().__init__()

        with self.init_scope():
            self.feature_extractor = ResNet(kwargs.pop('num_layers', 18))

        self.visual_backprop = VisualBackprop()
        self.visual_backprop_anchors = []
        self.out_size = out_size

        self.rotation_dropout_params = [1, 0, 1, 0, 1, 1]
        self.translation_dropout_params = [1, 1, 0, 1, 1, 0]

    @maybe_copy
    def __call__(self, images):
        self.visual_backprop_anchors.clear()
        h = self.feature_extractor(images)
        self.visual_backprop_anchors.append(h)

        batch_size = len(h)
        transform_params = self.get_transform_params(h)

        boxes = F.spatial_transformer_grid(transform_params, self.out_size)

        expanded_images = F.broadcast_to(F.expand_dims(images, axis=1), (batch_size, self.num_bboxes_to_localize) + images.shape[1:])
        expanded_images = F.reshape(expanded_images, (-1,) + expanded_images.shape[2:])
        rois = F.spatial_transformer_sampler(expanded_images, boxes)

        rois = F.reshape(rois, (batch_size, self.num_bboxes_to_localize, images.shape[1], self.out_size.height, self.out_size.width))
        boxes = F.reshape(boxes, (batch_size, self.num_bboxes_to_localize, 2, self.out_size.height, self.out_size.width))

        # return shapes:
        # 1. batch_size, num_bboxes, num_channels, (out-)height, (out-)width
        # 2. batch_size, num_bboxes, 2, (out-)height, (out-)width
        return rois, boxes

    def get_transform_params(self, features):
        raise NotImplementedError

    @maybe_copy
    def predict(self, images, return_visual_backprop=False):
        with cuda.Device(self._device_id):
            if isinstance(images, list):
                images = [self.xp.array(image) for image in images]
                images = self.xp.stack(images, axis=0)

            visual_backprop = None
            with chainer.using_config('train', False):
                roi, bbox = self(images)
                rois = [roi]
                bboxes = [bbox]
                if return_visual_backprop:
                    if not hasattr(self, 'visual_backprop'):
                        self.visual_backprop = VisualBackprop()
                    visual_backprop = self.visual_backprop.perform_visual_backprop(self.visual_backprop_anchors[0])

        bboxes = F.stack(bboxes, axis=1)
        bboxes = F.reshape(bboxes, (-1,) + bboxes.shape[2:])
        rois = F.stack(rois, axis=1)
        rois = F.reshape(rois, (-1,) + rois.shape[2:])

        return rois, bboxes, visual_backprop

    def virtual_box_number_increase(self, boxes, image_shape):
        image_shape = Size(*image_shape)
        offset_boxes = []
        box_offset_bounds = self.box_offset_side_length // 2
        x_box_shifts = self.xp.random.randint(1, 20, size=(self.box_offset_side_length, self.box_offset_side_length))
        y_box_shifts = self.xp.random.randint(1, 20, size=(self.box_offset_side_length, self.box_offset_side_length))
        for i in range(box_offset_bounds, box_offset_bounds + 1):
            for j in range(box_offset_bounds, box_offset_bounds + 1):
                x_shift = boxes[:, 0, :, :] + j * (x_box_shifts[i, j] / image_shape.width)
                y_shift = boxes[:, 1, :, :] + i * (y_box_shifts[i, j] / image_shape.height)
                offset_boxes.append(F.stack([x_shift, y_shift], axis=1))
        return F.stack(offset_boxes, axis=1)
