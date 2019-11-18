import numpy
import re
from chainer import reporter
from chainer.backends import cuda

import chainer.functions as F

from evaluation.custom_mean_evaluator import CustomMeanEvaluator
from evaluation.rotation_detection_evaluator import RotationMAPEvaluator
from image_manipulation.image_masking import ImageMasker
from train_utils.tensorboard_utils import TensorboardEvaluator

num_word_kernel = cuda.cupy.ElementwiseKernel(
    "raw T wordPrediction, int32 blankLabelClass, int32 maxWordLength",
    "raw int32 outElement",
    """
        // if we did not predict a blank label, the output of our function will be the full length 
        outElement[i] = maxWordLength;
        
        for (int j=0; j<maxWordLength; ++j) {
        // determine whether we are at the end of the word, if so we save the current position as output
            if (wordPrediction[i * maxWordLength + j] == blankLabelClass) {
                outElement[i] = j;
                break;
            }
        }
    """,
    name="word_length_determination"
)


class TextRecognitionEvaluatorFunction(RotationMAPEvaluator):

    def __init__(self, localizer, recognizer, device, blank_label_class, char_map):
        self.localizer = localizer
        self.recognizer = recognizer
        self.device = device
        self.blank_label_class = blank_label_class
        self.char_map = char_map
        self.image_masker = ImageMasker(1)
        self.blank_label_pattern = re.compile(f"{self.char_map[str(self.blank_label_class)]}")

    def __call__(self, **kwargs):
        image = kwargs.pop('image', None)
        words = kwargs.pop('words', None)
        masks = kwargs.pop('masks', None)
        return_predictions = kwargs.pop('return_predictions', False)

        with cuda.Device(self.device):
            rois, bboxes = self.localizer.predict(image)[:2]
            predicted_words = self.recognizer.predict(rois).array
            self.xp = cuda.get_array_module(bboxes)
            batch_size, num_bboxes, num_channels, height, width = rois.shape
            rois = self.xp.reshape(rois.array, (-1, num_channels, height, width))
            bboxes = self.xp.reshape(bboxes.array, (-1, 2, height, width))

            self.calc_word_accuracy(predicted_words, words)

            # masks = self.ndarray_to_list(masks)
            # num_predicted_chars = self.get_number_of_predicted_characters(predicted_words)
            #
            # batch_size, num_predicted_masks, pred_masks = self.bboxes_to_masks(bboxes, image)
            # pred_masks = self.zero_results_without_expected_prediction(num_predicted_chars, pred_masks)
            # pred_masks = self.ndarray_to_list(pred_masks)
            #
            # predicted_scores = self.assessor.extract_iou_prediction(self.assessor(rois)).data.reshape(batch_size, num_predicted_masks)
            # predicted_scores = self.zero_results_without_expected_prediction(num_predicted_chars, predicted_scores)
            # predicted_scores = self.ndarray_to_list(predicted_scores)
            # pred_masks, predicted_scores = self.perform_nms(batch_size, bboxes, num_predicted_masks, pred_masks, predicted_scores)

            # ious = self.xp.concatenate(self.calculate_iou(pred_masks, masks))
            # mean_iou = float(self.xp.sum(ious) / len(ious))
            # reporter.report({'mean_iou': mean_iou})

            # result = self.calculate_map(pred_masks, predicted_scores, masks)
            # reporter.report({'map': result['map']})

        if return_predictions:
            return rois, bboxes, predicted_words

    def calc_word_accuracy(self, predicted_words, words, strip_non_alphanumeric_predictions=False):
        num_correct_words = 0
        num_case_insensitive_correct_words = 0
        num_words = 0
        num_images = 0
        num_correct_lines = 0
        num_case_insensitive_correct_lines = 0
        num_correct_chars = 0
        num_chars = 0

        for predicted_image, gt_image in zip(predicted_words, words):
            num_correct_words_per_image = 0
            num_correct_case_insensitive_words_per_image = 0

            for predicted_word, gt_word in zip(predicted_image, gt_image):
                predicted_word = ''.join(self.char_map[str(int(char))] for char in predicted_word)
                predicted_word = re.sub(self.blank_label_pattern, '', predicted_word)
                gt_word = ''.join(self.char_map[str(int(char))] for char in gt_word)
                gt_word = re.sub(self.blank_label_pattern, '', gt_word)

                if strip_non_alphanumeric_predictions:
                    predicted_word = re.sub(r'[\W_]', '', predicted_word)

                num_correct_chars += sum(predicted_char == gt_char for predicted_char, gt_char in zip(predicted_word, gt_word))
                num_chars += len(gt_word)

                if predicted_word == gt_word:
                    num_correct_words_per_image += 1
                if predicted_word.lower() == gt_word.lower():
                    num_correct_case_insensitive_words_per_image += 1
                num_words += 1

            if num_correct_words_per_image == len(gt_image):
                num_correct_lines += 1
            if num_correct_case_insensitive_words_per_image == len(gt_image):
                num_case_insensitive_correct_lines += 1

            num_correct_words += num_correct_words_per_image
            num_case_insensitive_correct_words += num_correct_case_insensitive_words_per_image
            num_images += 1

        reporter.report({
            "correct_chars": num_correct_chars,
            "correct_words": num_correct_words,
            "correct_lines": num_correct_lines,
            "case_insensitive_correct_words": num_case_insensitive_correct_words,
            "case_insensitive_correct_lines": num_case_insensitive_correct_lines,
            "num_chars": num_chars,
            "num_images": num_images,
            "num_words": num_words
        })

    def zero_results_without_expected_prediction(self, word_lengths, predicted_masks):
        for word_length, predicted_mask in zip(word_lengths, predicted_masks):
            predicted_mask[word_length:] = 0
        return predicted_masks

    def get_number_of_predicted_characters(self, text_prediction):
        text_prediction = self.xp.squeeze(text_prediction)
        if self.xp == numpy:
            return self.get_number_of_predicted_characters_cpu(text_prediction)
        else:
            return self.get_number_of_predicted_characters_gpu(text_prediction)

    def get_number_of_predicted_characters_cpu(self, text_prediction):
        text_prediction = text_prediction.squeeze()
        num_predicted_characters = [text_prediction.shape[1]] * len(text_prediction)
        for batch_idx, prediction in enumerate(text_prediction):
            for i, char in enumerate(prediction):
                if int(char) == self.blank_label_class:
                    num_predicted_characters[batch_idx] = i
                    break
        return self.xp.array(num_predicted_characters)

    def get_number_of_predicted_characters_gpu(self, text_prediction):
        return num_word_kernel(
            text_prediction,
            self.blank_label_class,
            text_prediction.shape[1],
            size=len(text_prediction),
        )


class TextRecognitionTestFunction(TextRecognitionEvaluatorFunction):

    def __init__(self, *args, **kwargs):
        self.only_return_best_result = kwargs.pop('return_best_result', True)
        self.strip_non_alphanumeric_predictions = kwargs.pop('strip_non_alpha_numeric_predictions', False)
        super().__init__(*args, **kwargs)

    def determine_best_prediction_indices(self, raw_classification_result):
        distribution = F.softmax(raw_classification_result, axis=3)
        predicted_classes = F.argmax(distribution, axis=3)

        scores = []
        for i, image in enumerate(predicted_classes):
            means = []
            for j, image_variant in enumerate(image):
                num_predictions = len([prediction for prediction in image_variant if prediction.array != self.blank_label_class])
                probs = F.max(distribution[i, j, :num_predictions], axis=1).array
                means.append(self.xp.mean(probs))
            means = self.xp.stack(means, axis=0)
            scores.append(means)
        scores = self.xp.stack(scores, axis=0)
        # scores = F.sum(F.max(F.softmax(raw_classification_result, axis=3), axis=3), axis=2)
        best_indices = F.argmax(scores, axis=1).array
        best_indices = best_indices[:, self.xp.newaxis]
        return best_indices, scores

    def __call__(self, **kwargs):
        image = kwargs.pop('image', None)
        words = kwargs.pop('words', None)
        return_predictions = kwargs.pop('return_predictions', False)

        batch_size, images_per_image, num_channels, height, width = image.shape
        image = self.xp.reshape(image, (-1, num_channels, height, width))

        with cuda.Device(self.device):
            rois, bboxes = self.localizer.predict(image)[:2]
            predicted_words, raw_classification_result = self.recognizer.predict(rois, return_raw_classification_result=True)
            predicted_words = F.reshape(predicted_words, (batch_size, images_per_image) + predicted_words.shape[1:])
            raw_classification_result = F.reshape(
                raw_classification_result,
                (batch_size, images_per_image) + raw_classification_result.shape[1:]
            )

            best_indices, scores = self.determine_best_prediction_indices(raw_classification_result)
            chosen_indices = best_indices
            self.calc_word_accuracy(
                self.xp.concatenate([predicted_words[i, best_indices[i]].array for i in range(batch_size)], axis=0),
                words,
                self.strip_non_alphanumeric_predictions,
            )
            if not self.only_return_best_result:
                best_indices = self.xp.arange(images_per_image)[None, ...]
                best_indices = self.xp.tile(best_indices, (batch_size, 1))
            predicted_words = self.xp.stack([predicted_words[i, best_indices[i]].array for i in range(batch_size)], axis=0)

        if return_predictions:
            rois = F.reshape(rois, (batch_size, images_per_image) + rois.shape[1:])
            bboxes = F.reshape(bboxes, (batch_size, images_per_image) + bboxes.shape[1:])
            rois = self.xp.stack([rois[i, best_indices[i]].array for i in range(batch_size)], axis=0)
            bboxes = self.xp.stack([bboxes[i, best_indices[i]].array for i in range(batch_size)], axis=0)
            return rois, bboxes, predicted_words, best_indices, chosen_indices, scores


class TextRecognitionEvaluator(CustomMeanEvaluator):

    def calculate_mean_of_summary(self, summary):
        num_correct_words = summary._summaries["correct_words"]._x
        num_correct_case_insensitive_words = summary._summaries["case_insensitive_correct_words"]._x
        num_words = summary._summaries["num_words"]._x
        num_correct_lines = summary._summaries["correct_lines"]._x
        num_correct_case_insensitive_lines = summary._summaries["case_insensitive_correct_lines"]._x
        num_images = summary._summaries["num_images"]._x
        num_chars = summary._summaries["num_chars"]._x
        num_correct_chars = summary._summaries["num_correct_chars"]._x

        word_accuracy = num_correct_words / num_words
        line_accuracy = num_correct_lines / num_images
        case_insensitive_word_accuracy = num_correct_case_insensitive_words / num_words
        case_insensitive_line_accuracy = num_correct_case_insensitive_lines / num_images
        char_accuracy = num_correct_chars / num_chars

        return {
            "char_accuracy": char_accuracy,
            "word_accuracy": word_accuracy,
            "line_accuracy": line_accuracy,
            "case_insensitive_word_accuracy": case_insensitive_word_accuracy,
            "case_insensitive_line_accuracy": case_insensitive_line_accuracy,
        }


class TextRecognitionTensorboardEvaluator(TextRecognitionEvaluator, TensorboardEvaluator):

    def __init__(self, *args, **kwargs):
        super(TextRecognitionEvaluator, self).__init__(*args, **kwargs)
