from collections import namedtuple

import chainer.functions as F
import random
from chainer import Variable
from chainercv.utils import bbox_iou

Size = namedtuple('Size', ['height', 'width'])


def random_pairs(predicted_boxes):
    def pop_random(lst):
        return lst.pop(random.randrange(len(lst)))

    while len(predicted_boxes) > 1:
        box_1 = pop_random(predicted_boxes)
        box_2 = pop_random(predicted_boxes)
        yield box_1, box_2


class IOUCalculator:

    def __init__(self, xp, converter, device):
        self.xp = xp
        self.converter = converter
        self.device = device

    def overlap(self, x1, w1, x2, w2):
        return self.xp.maximum(self.xp.zeros_like(x1), self.xp.minimum(x1 + w1, x2 + w2) - self.xp.maximum(x1, x2))

    def intersection(self, bbox1, bbox2):
        width_overlap = self.overlap(bbox1[:, 0], bbox1[:, 2] - bbox1[:, 0], bbox2[:, 0], bbox2[:, 2] - bbox2[:, 0])
        height_overlap = self.overlap(bbox1[:, 1], bbox1[:, 3] - bbox1[:, 1], bbox2[:, 1], bbox2[:, 3] - bbox2[:, 1])

        overlaps = width_overlap * height_overlap
        overlaps = self.xp.maximum(overlaps, self.xp.zeros_like(overlaps))
        return overlaps

    def union(self, box1, box2, intersection_area=None):
        if intersection_area is None:
            intersection_area = self.intersection(box1, box2)
        union = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) + (box2[:, 2] - box2[:, 0]) * (box1[:, 3] - box1[:, 1]) - intersection_area
        return union

    def calc_bboxes(self, predicted_bboxes, image_size, out_size):
        bboxes = []
        for bbox in self.xp.split(predicted_bboxes, len(predicted_bboxes)):
            bbox = self.xp.squeeze(bbox, axis=0)
            bbox[...] = (bbox[...] + 1) / 2
            bbox[0, :] *= image_size.width
            bbox[1, :] *= image_size.height

            x = self.xp.clip(bbox[0, :].reshape(out_size), 0, image_size.width)
            y = self.xp.clip(bbox[1, :].reshape(out_size), 0, image_size.height)

            top_left = (x[0, 0], y[0, 0])
            top_right = (x[0, -1], y[0, -1])
            bottom_left = (x[-1, 0], y[-1, 0])
            bottom_right = (x[-1, -1], y[-1, -1])
            bboxes.append(self.xp.array(
                (
                    min(top_left[0], bottom_right[0]),
                    min(top_left[1], bottom_right[1]),
                    max(top_left[0], bottom_right[0]),
                    max(top_left[1], bottom_right[1]),
                ),
                dtype=self.xp.int32)
            )
        return self.converter(bboxes, self.device)

    def calc_iou(self, gt_bboxes, predicted_bboxes, image_size, out_size, xp=None):
        if xp is not None:
            self.xp = xp
        predicted_bboxes = self.calc_bboxes(predicted_bboxes, image_size, out_size)
        if gt_bboxes.size > predicted_bboxes.size:
            decrease_factor = predicted_bboxes.size / gt_bboxes.size
            assert gt_bboxes.shape[1] % decrease_factor**-1 == 0, "we have more gt bboxes than predicted, but is not possible to correctly slice them"
            gt_bboxes = gt_bboxes[:, :int(gt_bboxes.shape[1] * decrease_factor), ...]

        gt_bboxes = self.xp.reshape(gt_bboxes, (-1, gt_bboxes.shape[-1]))
        intersection_areas = self.intersection(gt_bboxes, predicted_bboxes)
        unions = self.union(gt_bboxes, predicted_bboxes, intersection_area=intersection_areas)
        ious = intersection_areas / unions

        return sum(ious) / max(len(intersection_areas), 1)


class SmoothIOUCalculator:

    def __init__(self, xp):
        self.xp = xp

    def overlap(self, x1, w1, x2, w2):
        return F.maximum(self.xp.zeros_like(x1), F.minimum(x1 + w1, x2 + w2) - F.maximum(x1, x2))

    def intersection(self, bbox1, bbox2):
        width_overlap = self.overlap(bbox1[:, 0], bbox1[:, 2] - bbox1[:, 0], bbox2[:, 0], bbox2[:, 2] - bbox2[:, 0])
        height_overlap = self.overlap(bbox1[:, 1], bbox1[:, 3] - bbox1[:, 1], bbox2[:, 1], bbox2[:, 3] - bbox2[:, 1])

        overlaps = width_overlap * height_overlap
        # overlaps = F.maximum(overlaps, self.xp.zeros_like(overlaps))
        return overlaps

    def union(self, box1, box2, intersection_area=None):
        if intersection_area is None:
            intersection_area = self.intersection(box1, box2)
        union = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) + (box2[:, 2] - box2[:, 0]) * (box1[:, 3] - box1[:, 1]) - intersection_area
        return union

    def smallest_area(self, box1, box2):
        area_box_1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area_box_2 = (box2[:, 2] - box2[:, 0]) * (box1[:, 3] - box1[:, 1])
        smallest_area = F.where(area_box_1.data < area_box_2.data, area_box_1, area_box_2)
        return smallest_area

    def calc_bboxes(self, predicted_bboxes, image_size, out_size):
        predicted_bboxes = (predicted_bboxes + 1) / 2
        x_points = predicted_bboxes[:, 0, ...] * image_size.width
        y_points = predicted_bboxes[:, 1, ...] * image_size.height
        top_left_x = F.get_item(x_points, [..., 0, 0])
        top_left_y = F.get_item(y_points, [..., 0, 0])
        bottom_right_x = F.get_item(x_points, [..., out_size.height - 1, out_size.width - 1])
        bottom_right_y = F.get_item(y_points, [..., out_size.height - 1, out_size.width - 1])

        bboxes = F.stack(
            [
                F.minimum(top_left_x, bottom_right_x),
                F.minimum(top_left_y, bottom_right_y),
                F.maximum(top_left_x, bottom_right_x),
                F.maximum(top_left_y, bottom_right_y),
            ],
            axis=1
        )
        return bboxes


class LossCalculator:

    def __init__(self, xp):
        self.xp = xp

    def get_corners(self, grids, image_size, scale_to_image_size=True):
        _, _, height, width = grids.shape
        if scale_to_image_size:
            grids = (grids + 1) / 2
            x_points = grids[:, 0, ...] * image_size.width
            y_points = grids[:, 1, ...] * image_size.height
        else:
            x_points = grids[:, 0, ...]
            y_points = grids[:, 1, ...]

        top_left_x = F.get_item(x_points, [..., 0, 0])
        top_left_y = F.get_item(y_points, [..., 0, 0])
        top_right_x = F.get_item(x_points, [..., 0, width - 1])
        top_right_y = F.get_item(y_points, [..., 0, width - 1])
        bottom_left_x = F.get_item(x_points, [..., height - 1, 0])
        bottom_left_y = F.get_item(y_points, [..., height - 1, 0])
        bottom_right_x = F.get_item(x_points, [..., height - 1, width - 1])
        bottom_right_y = F.get_item(y_points, [..., height - 1, width - 1])
        return top_left_x, top_right_x, bottom_left_x, bottom_right_x, top_left_y, top_right_y, bottom_left_y, bottom_right_y

    def calc_loss(self, grids, image_size, **kwargs):
        raise NotImplementedError


class DirectionLossCalculator(LossCalculator):

    def calc_loss(self, grids, image_size, **kwargs):
        normalize = kwargs.get('normalize', True)
        top_left_x, top_right_x, _, _, top_left_y, _, bottom_left_y, _ = self.get_corners(grids, image_size)

        # penalize upside down images
        distance = top_left_y - bottom_left_y
        up_down_loss = F.maximum(distance, self.xp.zeros_like(distance.array))
        if normalize:
            up_down_loss = F.sum(up_down_loss)

        # penalize images that are vertically mirrored
        distance = top_left_x - top_right_x
        left_right_loss = F.maximum(distance, self.xp.zeros_like(distance.array))
        if normalize:
            left_right_loss = F.sum(left_right_loss)

        return up_down_loss + left_right_loss


class MinAreaLossCalculator(LossCalculator):

    def calc_loss(self, grids, image_size, **kwargs):
        """
            Calculate a loss based on the expected grid size. Penalize all predicted grids, where the area of the grid
            is smaller than the area of the crop area
        """
        top_left_x, top_right_x, _, _, top_left_y, _, bottom_left_y, _ = self.get_corners(grids, image_size)

        grid_widths = top_right_x - top_left_x
        grid_heights = bottom_left_y - top_left_y
        expected_width = self.xp.full_like(grid_widths.array, grids.shape[-1], dtype=grid_widths.dtype)
        expected_height = self.xp.full_like(grid_heights.array, grids.shape[2], dtype=grid_heights.dtype)

        width_loss = F.maximum(self.xp.zeros_like(grid_widths.array), expected_width - grid_widths)
        height_loss = F.maximum(self.xp.zeros_like(grid_heights.array), expected_height - grid_heights)

        return sum(width_loss) + sum(height_loss)


class MaxAreaLossCalculator(LossCalculator):

    def calc_loss(self, grids, image_size, **kwargs):
        top_left_x, top_right_x, _, _, top_left_y, _, bottom_left_y, _ = self.get_corners(grids, image_size)

        grid_widths = top_right_x - top_left_x
        grid_heights = bottom_left_y - top_left_y
        expected_width = self.xp.full_like(grid_widths, image_size.width, dtype=grid_widths.dtype)
        expected_height = self.xp.full_like(grid_heights, image_size.height, dtype=grid_heights.dtype)

        width_loss = F.maximum(self.xp.zeros_like(grid_widths), grid_widths - expected_width)
        height_loss = F.maximum(self.xp.zeros_like(grid_heights), grid_heights - expected_height)

        return sum(width_loss) + sum(height_loss)


class AspectRatioLossCalculator(LossCalculator):

    def get_bbox_side_lengths(self, grids, image_size):
        x0, x1, x2, _, y0, y1, y2, _ = self.get_corners(grids, image_size)

        width = F.sqrt(
            F.square(x1 - x0) + F.square(y1 - y0)
        )

        height = F.sqrt(
            F.square(x2 - x0) + F.square(y2 - y0)
        )
        return width, height

    def calc_loss(self, grids, image_size, **kwargs):
        normalize = kwargs.get('normalize', True)
        width, height = self.get_bbox_side_lengths(grids, image_size)

        # penalize aspect ratios that are higher than wide, and penalize aspect ratios that are tooo wide
        aspect_ratio = height / F.maximum(width, self.xp.ones_like(width))
        # do not give an incentive to bboxes with a width that is 2x the height of the box
        aspect_loss = F.maximum(aspect_ratio - 0.5, self.xp.zeros_like(aspect_ratio))

        return F.mean(aspect_loss)


class TransformParameterRegressionLossCalculator(LossCalculator):

    def calc_loss(self, image_size, predicted_grids, gt_bbox_points, objectness_scores, normalize=True):
        predicted_bbox_points = self.get_corners(predicted_grids, image_size, scale_to_image_size=False)

        # 1. transform box coordinates to aabb coordinates for determination of iou
        predicted_bbox_points = predicted_bbox_points[0], predicted_bbox_points[4], predicted_bbox_points[3], predicted_bbox_points[7]
        predicted_bbox_points = F.stack(predicted_bbox_points, axis=1)

        # 2. find best prediction area for each gt bbox
        gt_bboxes_to_use_for_loss = []
        positive_anchor_indices = self.xp.empty((0,), dtype=self.xp.int32)
        not_contributing_anchors = self.xp.empty((0,), dtype=self.xp.int32)
        for index, gt_bbox in enumerate(gt_bbox_points):
            # determine which bboxes are positive boxes as they have high iou with gt and also which bboxes are negative
            # this is also used to train objectness classification
            gt_bbox = self.xp.tile(gt_bbox[None, ...], (len(predicted_bbox_points), 1))

            ious = bbox_iou(gt_bbox, predicted_bbox_points.data)
            positive_boxes = self.xp.where((ious[0] >= 0.7))
            not_contributing_boxes = self.xp.where(self.xp.logical_and(0.3 < ious[0], ious[0] < 0.7))
            if len(positive_boxes[0]) == 0:
                best_iou_index = ious[0, :].argmax()
                positive_anchor_indices = self.xp.concatenate((positive_anchor_indices, best_iou_index[None, ...]), axis=0)
                gt_bboxes_to_use_for_loss.append(gt_bbox[0])
            else:
                positive_anchor_indices = self.xp.concatenate((positive_anchor_indices, positive_boxes[0]), axis=0)
                gt_bboxes_to_use_for_loss.extend(gt_bbox[:len(positive_boxes[0])])
            not_contributing_anchors = self.xp.concatenate((not_contributing_anchors, not_contributing_boxes[0]), axis=0)

        if len(gt_bboxes_to_use_for_loss) == 0:
            return Variable(self.xp.array(0, dtype=predicted_grids.dtype))

        gt_bboxes_to_use_for_loss = F.stack(gt_bboxes_to_use_for_loss)

        # filter predicted bboxes and only keep bboxes from those regions that actually contain a bbox
        predicted_bbox_points = F.get_item(predicted_bbox_points, positive_anchor_indices)

        # 3. calculate L1 loss for bbox regression
        loss = F.huber_loss(
            predicted_bbox_points,
            gt_bboxes_to_use_for_loss,
            1
        )

        # 4. calculate objectness loss
        objectness_labels = self.xp.zeros(len(objectness_scores), dtype=self.xp.int32)
        objectness_labels[not_contributing_anchors] = -1
        objectness_labels[positive_anchor_indices] = 1

        objectness_loss = F.softmax_cross_entropy(
            objectness_scores,
            objectness_labels,
            ignore_label=-1,
        )

        return F.mean(loss), objectness_loss


class OutOfImageLossCalculator(LossCalculator):

    def calc_loss(self, grids, image_size, **kwargs):
        normalize = kwargs.get('normalize', True)
        corner_coordinates = self.get_corners(grids, image_size, scale_to_image_size=False)
        # determine whether a point is out of the image, image range is [-1, 1]
        # everything outside of this increases the loss!
        bbox = F.concat(corner_coordinates, axis=0)
        top_loss = bbox + 1.5
        bottom_loss = bbox - 1.5

        # do not penalize anything inside the image
        top_loss = F.absolute(F.minimum(top_loss, self.xp.zeros_like(top_loss.array)))
        top_loss = F.reshape(top_loss, (len(corner_coordinates), -1))
        bottom_loss = F.maximum(bottom_loss, self.xp.zeros_like(bottom_loss.array))
        bottom_loss = F.reshape(bottom_loss, (len(corner_coordinates), -1))

        loss = F.sum(F.concat([top_loss, bottom_loss], axis=0), axis=0)
        if normalize:
            loss = F.sum(loss)
        return loss


class PairWiseOverlapLossCalculator(LossCalculator):

    def overlap(self, x1, w1, x2, w2):
        return F.maximum(self.xp.zeros_like(x1.array), F.minimum(x1 + w1, x2 + w2) - F.maximum(x1, x2))

    def intersection(self, bbox1, bbox2):
        width_overlap = self.overlap(bbox1[:, 0, 0], bbox1[:, 1, 0] - bbox1[:, 0, 0], bbox2[:, 0, 0], bbox2[:, 1, 0] - bbox2[:, 0, 0])
        height_overlap = self.overlap(bbox1[:, 1, 1], bbox1[:, 3, 1] - bbox1[:, 1, 1], bbox2[:, 1, 1], bbox2[:, 3, 1] - bbox2[:, 1, 1])

        intersection_areas = width_overlap * height_overlap
        return intersection_areas

    def calc_loss(self, grids, image_size, **kwargs):
        num_grids = kwargs.pop('num_grids')
        corners = self.get_corners(grids, image_size, scale_to_image_size=False)
        top_left_x, top_right_x, bottom_left_x, bottom_right_x, top_left_y, top_right_y, bottom_left_y, bottom_right_y = corners
        corner_coordinates = F.stack(
            [
                top_left_x,
                top_left_y,
                top_right_x,
                top_right_y,
                bottom_left_x,
                bottom_left_y,
                bottom_right_x,
                bottom_right_y
            ],
            axis=1
        )
        corner_coordinates = F.reshape(corner_coordinates, (-1, num_grids, 4, 2))

        grids = F.separate(corner_coordinates, axis=1)
        intersection_areas = []
        for box1, box2 in zip(grids, grids[1:]):
            intersection_areas.append(self.intersection(box1, box2))

        loss = F.sum(F.stack(intersection_areas, axis=1))
        return loss
