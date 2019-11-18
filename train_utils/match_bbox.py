from collections import namedtuple

import chainer.functions as F
import math

from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.pooling.pooling_2d import Pooling2D
from chainer.utils import conv

ReceptiveField = namedtuple('ReceptiveField', ['n', 'jump', 'size', 'start'])


class LayerExtractor:

    def __init__(self):
        self.layers = []

    def traverse_computational_graph(self, node):
        if isinstance(node, Convolution2DFunction) or isinstance(node, Pooling2D):
            self.layers.append(node)

        if node.inputs[0].creator is None:
            return
        return self.traverse_computational_graph(node.inputs[0].creator)

    def find_conv_and_pool_layers(self, node):
        self.layers.clear()
        self.traverse_computational_graph(node.creator)
        return reversed(self.layers)


def get_bbox_corners(grids, image_size):
        _, _, height, width = grids.shape
        grids = (grids + 1) / 2
        x_points = grids[:, 0, ...] * image_size.width
        y_points = grids[:, 1, ...] * image_size.height
        x_points = F.clip(x_points, 0., float(image_size.width))
        y_points = F.clip(y_points, 0., float(image_size.height))
        top_left_x = F.get_item(x_points, [..., 0, 0])
        top_left_y = F.get_item(y_points, [..., 0, 0])
        bottom_right_x = F.get_item(x_points, [..., height - 1, width - 1])
        bottom_right_y = F.get_item(y_points, [..., height - 1, width - 1])
        return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def get_aabb_corners(grids, image_size):
    _, _, height, width = grids.shape
    grids = (grids + 1) / 2
    x_points = grids[:, 0, ...] * image_size.width
    y_points = grids[:, 1, ...] * image_size.height
    x_points = F.clip(x_points, 0., float(image_size.width))
    y_points = F.clip(y_points, 0., float(image_size.height))
    top_left_x = F.get_item(x_points, [..., 0, 0])
    top_left_y = F.get_item(y_points, [..., 0, 0])
    top_right_x = F.get_item(x_points, [..., 0, width - 1])
    top_right_y = F.get_item(y_points, [..., 0, width - 1])
    bottom_right_x = F.get_item(x_points, [..., height - 1, width - 1])
    bottom_right_y = F.get_item(y_points, [..., height - 1, width - 1])
    bottom_left_x = F.get_item(x_points, [..., height - 1, 0])
    bottom_left_y = F.get_item(y_points, [..., height - 1, 0])

    top_left_x_aabb = F.minimum(top_left_x, bottom_left_x)
    top_left_y_aabb = F.minimum(top_left_y, top_right_y)
    bottom_right_x_aabb = F.maximum(top_right_x, bottom_right_x)
    bottom_right_y_aabb = F.maximum(bottom_left_y, bottom_right_y)

    return top_left_y_aabb, top_left_x_aabb, bottom_right_y_aabb, bottom_right_x_aabb


def bbox_coords_to_feature_coords(bbox, receptive_field_width, receptive_field_height, xp):
    feature_map_width, jump_width, receptive_field_width, start_index_width = receptive_field_width
    feature_map_height, jump_height, receptive_field_height, start_index_height = receptive_field_height

    top_left_x = xp.clip(xp.floor((bbox[0].data - start_index_width) / jump_width), 0, feature_map_width + 1)
    top_left_y = xp.clip(xp.floor((bbox[1].data - start_index_height) / jump_height), 0, feature_map_height + 1)
    bottom_right_x = xp.clip(xp.ceil((bbox[2].data - start_index_width) / jump_width), 0, feature_map_width + 1)
    bottom_right_y = xp.clip(xp.ceil((bbox[3].data - start_index_height) / jump_height), 0, feature_map_height + 1)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def calculate_receptive_fields(layers, image_size):
    # code is based on this blog post:
    # https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

    def get_receptive_field(receptive_field_info, kernel, padding, stride):
        n_in, j_in, r_in, start_in = receptive_field_info
        n_out = conv.get_conv_outsize(n_in, kernel, stride, padding, cover_all=False)

        j_out = j_in * stride
        r_out = r_in + (kernel - 1) * j_in
        start_out = start_in + ((kernel - 1) / 2 - padding) * j_in
        return n_out, j_out, r_out, start_out

    receptive_field_info_width = [image_size.width, 1, 1, 0.5]
    receptive_field_info_height = [image_size.height, 1, 1, 0.5]

    for layer in layers:
        if isinstance(layer, Pooling2D):
            kernel_h, kernel_w = layer.kh, layer.kw
        else:
            kernel_h, kernel_w = layer.inputs[1].shape[-2:]
        stride_h = layer.sy
        stride_w = layer.sx
        pad_h = layer.ph
        pad_w = layer.pw

        receptive_field_info_width = get_receptive_field(receptive_field_info_width, kernel_w, pad_w, stride_w)
        receptive_field_info_height = get_receptive_field(receptive_field_info_height, kernel_h, pad_h, stride_h)

    return ReceptiveField._make(receptive_field_info_width), ReceptiveField._make(receptive_field_info_height)






