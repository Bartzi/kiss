import chainercv
import numpy

from chainer import reporter
from chainer.backends import cuda
from chainercv.utils import mask_iou

from common.utils import Size
from image_manipulation.image_masking import ImageMasker


class RotationMAPEvaluator:

    def __init__(self, link, device, assessor=None):
        self.link = link
        self.device = device
        self.assessor = assessor
        self.image_masker = ImageMasker(1)

    def __call__(self, *inputs):
        images, labels = inputs[:2]
        with cuda.Device(self.device):
            rois, bboxes = self.link.predict(images)[:2]

            self.xp = cuda.get_array_module(bboxes)
            bboxes = bboxes.data
            labels = self.ndarray_to_list(labels)

            batch_size, num_predicted_masks, pred_masks = self.bboxes_to_masks(bboxes, images)
            pred_masks = self.ndarray_to_list(pred_masks)

            if self.assessor is not None:
                pred_scores = self.ndarray_to_list(
                    self.assessor.extract_iou_prediction(self.assessor(rois)).data.reshape(batch_size, num_predicted_masks)
                )
                pred_masks, pred_scores = self.perform_nms(batch_size, bboxes, num_predicted_masks, pred_masks, pred_scores)
            else:
                pred_scores = self.ndarray_to_list(numpy.ones((batch_size, num_predicted_masks)))

            ious = self.xp.concatenate(self.calculate_iou(pred_masks, labels))
            mean_iou = float(self.xp.sum(ious) / len(ious))
            reporter.report({'mean_iou': mean_iou})

            result = self.calculate_map(pred_masks, pred_scores, labels)
            reporter.report({'map': result['map']})

    def ndarray_to_list(self, array, axis=0):
        return [self.xp.squeeze(arr, axis=axis) for arr in self.xp.split(array, array.shape[axis], axis=axis)]

    def perform_nms(self, batch_size, bboxes, num_predicted_masks, pred_masks, pred_scores):
        nms_bboxes = self.prepare_non_maximum_suppression(bboxes)
        nms_bboxes = self.xp.reshape(nms_bboxes, (batch_size, num_predicted_masks, -1))
        for i, (nms_bboxes_per_image, scores_per_image) in enumerate(zip(nms_bboxes, pred_scores)):
            bboxes_to_keep = chainercv.utils.non_maximum_suppression(
                nms_bboxes_per_image,
                0.7,
                score=scores_per_image
            )
            pred_masks[i] = pred_masks[i][bboxes_to_keep]
            pred_scores[i] = pred_scores[i][bboxes_to_keep]

        return pred_masks, pred_scores

    def calculate_map(self, pred_masks, pred_scores, labels):
        pred_masks = cuda.to_cpu(pred_masks)
        pred_scores = cuda.to_cpu(pred_scores)
        pred_labels = [numpy.zeros_like(pred_scores[i]) for i in range(len(pred_scores))]

        gt_masks = cuda.to_cpu(labels)
        gt_labels = [numpy.zeros(len(gt_mask), dtype=pred_labels[0].dtype) for gt_mask in gt_masks]

        return chainercv.evaluations.eval_instance_segmentation_voc(
            pred_masks,
            pred_labels,
            pred_scores,
            gt_masks,
            gt_labels
        )

    def calculate_iou(self, pred_masks, labels):
        best_ious = []
        for image_masks, gt_image_masks in zip(pred_masks, labels):
            ious = mask_iou(image_masks, gt_image_masks)
            best_ious.append(self.xp.max(ious, axis=1))

        return best_ious

    def bboxes_to_masks(self, bboxes, images):
        image_size = Size._make(images.shape[-2:])

        corners = self.image_masker.extract_corners(bboxes, self.xp)
        corners = self.image_masker.scale_bboxes(corners, image_size).astype(self.xp.int32)

        batch_size, _, height, width = images.shape
        pred_masks = self.image_masker.create_mask((len(bboxes), height, width), corners, self.xp)
        num_predicted_masks = len(pred_masks) // batch_size
        pred_masks = self.xp.reshape(pred_masks, (batch_size, num_predicted_masks, height, width))
        return batch_size, num_predicted_masks, pred_masks

    def prepare_non_maximum_suppression(self, bboxes):
        bbox_corners = self.image_masker.extract_corners(bboxes, self.xp)
        x_min = bbox_corners[:, 0, 0]
        y_min = bbox_corners[:, 0, 1]
        x_max = bbox_corners[:, 2, 0]
        y_max = bbox_corners[:, 2, 1]

        nms_bboxes = self.xp.stack([y_min, x_min, y_max, x_max], axis=1)
        return nms_bboxes
