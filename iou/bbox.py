from collections import namedtuple

import chainer
import math
import numpy as np
from PIL import Image, ImageDraw
from chainer.functions import spatial_transformer_grid, spatial_transformer_sampler
from functools import reduce
from shapely.geometry import Polygon, LineString
from shapely.ops import cascaded_union

AxisAlignedBBoxBase = namedtuple("BBox", ["left", "top", "right", "bottom"])


class AxisAlignedBBox(AxisAlignedBBoxBase):

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def area(self):
        return self.width * self.height

    def to_numpy(self):
        return np.array(self)

    def crop_from_image(self, image):
        return image.crop(self)

    def enlarge(self, image_size, width_enlargement=0.2, height_enlargement=0.05):
        extra_width = self.width * width_enlargement // 2
        extra_height = self.height * height_enlargement // 2

        new_bbox = AxisAlignedBBox(
            max(self.left - extra_width, 0),
            max(self.top - extra_height, 0),
            min(self.right + extra_width, image_size[0]),
            min(self.bottom + extra_height, image_size[1]),
        )

        return new_bbox

    def __add__(self, other):
        assert isinstance(other, Vec), "You can only add a vector to an aabb"
        return AxisAlignedBBox(
            self.left + other.x,
            self.top + other.y,
            self.right + other.x,
            self.bottom + other.y
        )

    def scale(self, scaling_factors):
        if not isinstance(scaling_factors, Vec):
            scaling_factors = Vec2(*scaling_factors)

        return AxisAlignedBBox(
            self.left * scaling_factors.x,
            self.top * scaling_factors.y,
            self.right * scaling_factors.x,
            self.bottom * scaling_factors.y,
        )


class ComparableMixin:
    def __eq__(self, other):
        return not self < other and not other < self

    def __ne__(self, other):
        return self < other or other < self

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not other < self

    def __lt__(self, other):
        raise NotImplementedError


Vec = namedtuple("Vec2", ["x", "y"])


class Vec2(Vec, ComparableMixin):

    def is_valid(self, maxima):
        return all(0 <= i <= m for i, m in zip(self, maxima))

    def normalize(self):
        return Vec2(
            self.x / self.length,
            self.y / self.length
        )

    @property
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def elementwise_max(self, other):
        return Vec2(
            self.x if self.x >= other.x else other.x,
            self.y if self.y >= other.y else other.y,
        )

    def elementwise_min(self, other):
        return Vec2(
            self.x if self.x <= other.x else other.x,
            self.y if self.y <= other.y else other.y,
        )

    def __add__(self, other):
        return Vec2(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other):
        if isinstance(other, Vec):
            return Vec2(
                self.x - other.x,
                self.y - other.y,
            )
        return Vec2(
            self.x - other,
            self.y - other
        )

    def __mul__(self, other):
        if isinstance(other, Vec):
            return Vec2(
                self.x * other.x,
                self.y * other.y,
            )
        return Vec2(
            self.x * other,
            self.y * other,
        )

    def __truediv__(self, other):
        if isinstance(other, Vec):
            return Vec2(
                self.x / other.x,
                self.y / other.y,
            )
        return Vec2(
            self.x / other,
            self.y / other
        )

    def __neg__(self):
        return Vec2(
            -self.x,
            -self.y
        )

    def to_int(self):
        return Vec2(
            int(self.x),
            int(self.y),
        )

    def __lt__(self, other):
        if isinstance(other, Vec):
            return self.length < other.length
        if isinstance(other, tuple):
            return self.x < other[0] and self.y < other[1]

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def angle_to(self, other):
        return self.dot(other) / (self.length * other.length + 1e-8)

    def normal(self):
        return Vec2(self.y, -self.x)

    def distance(self, other):
        return (other - self).length

    def to_normalized_coordinates(self, image_size):
        return Vec2(
            self.x / image_size[0],
            self.y / image_size[1]
        )


BBoxBase = namedtuple("BBox", ["top_left", "top_right", "bottom_right", "bottom_left", ])


class BBox(BBoxBase):

    def __new__(cls, *args,):
        int_args = [arg.to_int() for arg in args]
        return super().__new__(cls, *int_args)

    def __eq__(self, other):
        return all(self_corner == other_corner for self_corner, other_corner in zip(self, other))

    @staticmethod
    def merge(boxes):
        polygons = [box.to_polygon() for box in boxes]
        union = cascaded_union(polygons)
        union_aabb = union.envelope
        return BBox.from_axis_aligned_bbox(AxisAlignedBBox(*union_aabb.bounds))

    def is_valid(self, maxima):
        corners_valid = all(corner.is_valid(maxima) for corner in self)
        return corners_valid and self.area > 0

    def is_upright(self):
        return self.height > 0 and self.width > 0

    @property
    def x_length(self):
        return Vec2(self.top_right.x - self.top_left.x, self.top_right.y - self.top_left.y).length

    @property
    def y_length(self):
        return Vec2(self.bottom_left.x - self.top_left.x, self.bottom_left.y - self.top_left.y).length

    @property
    def width(self):
        most_left_point = min([self.top_left.x, self.bottom_left.x])
        most_right_point = max([self.top_right.x, self.bottom_right.x])
        return most_right_point - most_left_point

    @property
    def height(self):
        highest_point = min([self.top_left.y, self.top_right.y])
        lowest_point = max([self.bottom_left.y, self.bottom_right.y])
        return lowest_point - highest_point

    @property
    def area(self):
        return abs(self.width * self.height)

    @property
    def slant(self):
        top_vec = self.top_right - self.top_left
        left_vec = self.bottom_left - self.top_left
        return top_vec.angle_to(left_vec)

    @property
    def angle(self):
        top_vec = self.top_right - self.top_left
        x_axis = Vec2(1, 0)
        return x_axis.angle_to(top_vec)

    @property
    def center(self):
        # calculate the center of a box based on the diagonals
        top_left_bottom_right = LineString([self.bottom_right, self.top_left])
        bottom_left_top_right = LineString([self.top_right, self.bottom_left])
        intersection_point = top_left_bottom_right.intersection(bottom_left_top_right)
        return Vec2(x=intersection_point.x, y=intersection_point.y)

    @staticmethod
    def from_axis_aligned_bbox(bbox):
        return BBox(
            Vec2(bbox.left, bbox.top),
            Vec2(bbox.right, bbox.top),
            Vec2(bbox.right, bbox.bottom),
            Vec2(bbox.left, bbox.bottom),
        )

    @staticmethod
    def from_list(l):
        return BBox(*[Vec2._make(corner) for corner in l])

    @staticmethod
    def from_size_and_angles(start_point, width, height, rotation_angle, shear_angle):
        width_vector = Vec2(
            width * math.cos(math.radians(rotation_angle)),
            -(width * math.sin(math.radians(rotation_angle)))
        )
        height_vector = Vec2(
            height * math.cos(math.radians(rotation_angle - 90)),
            -(height * math.sin(math.radians(rotation_angle - 90)))
        )
        top_left = start_point
        top_right = top_left + width_vector
        bottom_right = top_right + height_vector
        bottom_left = bottom_right - width_vector

        box = BBox(top_left, top_right, bottom_right, bottom_left)
        shear_coefficient = 1 / math.tan((math.pi / 2) - math.radians(shear_angle))
        return box.shear(shear_coefficient)

    def iou(self, other):
        a = self.to_polygon()
        b = other.to_polygon()
        return a.intersection(b).area / max(a.union(b).area, 1)

    def shear(self, shear_corefficient):
        shear_matrix = np.array([[1, shear_corefficient], [0, 1]])
        sheared_points = [Vec2(*np.dot(shear_matrix, np.array(point))) for point in self]
        return BBox(*sheared_points)

    def to_polygon(self):
        return Polygon(self)

    def enlarge(self, image_size, width_enlargement=0.2, height_enlargement=0.05):
        extra_width = self.width * width_enlargement // 2
        extra_height = self.height * height_enlargement // 2

        width_vector = Vec2(self.top_right.x - self.top_left.x, self.top_right.y - self.top_left.y).normalize()
        height_vector = Vec2(self.bottom_right.x - self.top_right.x, self.bottom_right.y - self.top_right.y).normalize()

        top_left = self.top_left - width_vector * extra_width - height_vector * extra_height
        top_right = self.top_right + width_vector * extra_width - height_vector * extra_height
        bottom_right = self.bottom_right + width_vector * extra_width + height_vector * extra_height
        bottom_left = self.bottom_left - width_vector * extra_width + height_vector * extra_height

        top_left = Vec2(max(top_left.x, 0), max(top_left.y, 0))
        top_right = Vec2(min(top_right.x, image_size[0]), max(top_right.y, 0))
        bottom_right = Vec2(min(bottom_right.x, image_size[0]), min(bottom_right.y, image_size[1]))
        bottom_left = Vec2(max(bottom_left.x, 0), min(bottom_left.y, image_size[1]))

        new_bbox = BBox(
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        )

        return new_bbox

    def to_aabb(self):
        left = min(self.top_left.x, self.bottom_left.x, self.top_right.x, self.bottom_right.x)
        top = min(self.top_left.y, self.bottom_left.y, self.top_right.y, self.bottom_right.y)
        right = max(self.top_left.x, self.bottom_left.x, self.top_right.x, self.bottom_right.x)
        bottom = max(self.top_left.y, self.bottom_left.y, self.top_right.y, self.bottom_right.y)

        return AxisAlignedBBox(left, top, right, bottom)

    def get_crop_mask(self, image_data):
        mask_image = Image.new('L', (image_data.shape[1], image_data.shape[0]), 0)
        ImageDraw.Draw(mask_image).polygon(self, fill=1, outline=1)
        return np.array(mask_image)

    def crop_from_image(self, image, output_size=None, use_spatial_transformer=True):
        if output_size is None:
            output_size = (self.width, self.height)

        if use_spatial_transformer:
            image_array = np.asarray(image).transpose(2, 0, 1).astype(np.float32)
            crop_transform = self.get_affine_transform_params(image.size).astype(np.float32)
            transform_grid = spatial_transformer_grid(crop_transform[np.newaxis, ...], (output_size[1], output_size[0]))
            cropped_image = spatial_transformer_sampler(image_array[np.newaxis, ...], transform_grid).data[0]
            cropped_image = cropped_image.astype(np.uint8)

            cropped_image = Image.fromarray(cropped_image.transpose(1, 2, 0))
        else:
            cropped_image = image.crop(self.to_aabb())
            cropped_image = cropped_image.resize(output_size, Image.BILINEAR)

        return cropped_image

    def get_affine_transform_params(self, image_size, xp=np):
        # find transformation params using three known points (start points of untransformed image and points of our bbox
        # see https://stackoverflow.com/questions/22954239/given-three-points-compute-affine-transformation for a nice
        # explanation
        # we need to transform all values into the range [-1, 1] becaus we want to use chainers spatial transformer tool

        image_size = Vec2(*image_size)
        x_prime = (
            list(self.top_left / image_size * 2 - 1) +
            list(self.top_right / image_size * 2 - 1) +
            list(self.bottom_left / image_size * 2 - 1)
        )

        a = []
        for x, y in [(-1, -1), (1, -1), (-1, 1)]:  # top-left, top-right, bottom-left
            rows = xp.array([
                [x, y, 1, 0, 0, 0],
                [0, 0, 0, x, y, 1]
            ], dtype=chainer.get_dtype())
            a.append(rows)

        a = xp.concatenate(a, axis=0)
        A = xp.linalg.lstsq(a, xp.array(x_prime, dtype=chainer.get_dtype()))[0].reshape((2, 3))
        return A

    def intersects(self, other):
        return self.to_polygon().intersects(other.to_polygon())

    def intersects_enough(self, other, min_intersection_area):
        other = other.to_polygon()

        intersection = self.to_polygon().intersection(other)
        return intersection.area / max(other.area, 1) > min_intersection_area

    def intersection_ratio(self, other):
        other = other.to_polygon()
        return self.to_polygon().intersection(other).area / max(other.area, 1)

    def coverage_ratio(self, other):
        return self.intersection_ratio(other)

    def overcoverage_ratio(self, other):
        other = other.to_polygon()
        intersection = self.to_polygon().intersection(other)
        return (self.area - intersection.area) / self.area

    def multi_intersection_ratio(self, others):
        # return the ratio the current bbox intersects with all other boxes
        polygon_self = self.to_polygon()
        intersections = [polygon_self.intersection(other.to_polygon()) for other in others]
        union_of_intersections = reduce(lambda x, y: x.union(y), intersections[1:], intersections[0])
        return union_of_intersections.area / max(polygon_self.area, 1)

    def get_largest_iou(self, others):
        ious = [self.iou(other) for other in others]
        largest_iou = max(ious)
        corresponding_box = others[ious.index(largest_iou)]
        return largest_iou, corresponding_box

    def get_distance(self, other, image_size):
        # returns the distance of the center of the current box from the center of the other box
        other_center = other.center.to_normalized_coordinates(image_size)
        self_center = self.center.to_normalized_coordinates(image_size)
        return self_center.distance(other_center)

    def scale(self, scaling_factors):
        if not isinstance(scaling_factors, Vec):
            scaling_factors = Vec2(*scaling_factors)

        return BBox(*[corner * scaling_factors for corner in self])
