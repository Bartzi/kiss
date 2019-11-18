import argparse
import importlib
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import chainer
import chainer.functions as F
import matplotlib
from PIL import Image
from chainer.dataset import concat_examples

from common.datasets.text_recognition_eval_dataset import TextRecognitionEvaluationDataset
from evaluation.text_recognition_evaluator import TextRecognitionEvaluator, \
    TextRecognitionTestFunction
from insights.text_recognition_bbox_plotter import TextRecognitionBBoxPlotter
from train_utils.backup import restore_backup

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
from chainer import configuration
from chainer import reporter as reporter_module
from chainer.backends import cuda
from chainercv.utils import non_maximum_suppression
from tqdm import tqdm
from xml.dom import minidom

from insights.bbox_plotter import get_next_color
from train_utils.datatypes import Size
from train_utils.match_bbox import get_aabb_corners


class Evaluator:

    def __init__(self, args):
        self.args = args

        with open(os.path.join(args.model_dir, args.log_name)) as the_log_file:
            log_data = json.load(the_log_file)[0]

        self.image_size = Size(*log_data['image_size'])
        self.target_size = Size(*log_data['target_size'])

        if self.args.char_map is None:
            self.args.char_map = log_data['char_map']

        # Step 0 Build Dataset
        self.data_loader = TextRecognitionEvaluationDataset(
            self.image_size,
            npz_file=args.eval_gt,
            char_map=self.args.char_map,
            root=os.path.dirname(args.eval_gt),
            dtype=chainer.get_dtype(),
            transform_probability=0,
            keep_aspect_ratio=log_data.get('keep_aspect_ratio', True),
            image_mode=log_data.get('image_mode', 'RGB')
        )

        if hasattr(self.data_loader, 'num_chars'):
            self.data_loader.num_chars = log_data['num_words']
            self.data_loader.num_words = log_data['num_chars']
        else:
            self.data_loader.metadata['num_chars'] = log_data['num_words']
            self.data_loader.metadata['num_words'] = log_data['num_chars']

        if args.num_samples is not None:
            self.data_loader.shrink_dataset(args.num_samples)

        self.data_iterator = chainer.iterators.MultiprocessIterator(
            self.data_loader,
            args.batchsize,
            repeat=False,
            shuffle=False
        )

        # step 1 build network
        localizer_class = restore_backup(log_data['localizer'], args.model_dir)
        log_keys = ['use_group_norm', 'rotation_ratio', 'num_bboxes_to_localize', 'num_layers', 'rotation_dropout_ratio']
        localizer_args = {key: log_data[key] for key in log_keys if key in log_data}
        if 'num_bboxes_to_localize' not in localizer_args:
            localizer_args['num_bboxes_to_localize'] = self.data_loader.num_words_per_image
        if 'rotation_dropout_ratio' in localizer_args:
            localizer_args['dropout_ratio'] = localizer_args['rotation_dropout_ratio']

        try:
            self.localizer = localizer_class(
                self.target_size,
                **localizer_args
            )
        except (ValueError, KeyError) as e:
            print(e)
            print("Can not create localizer with standard args, falling back")
            self.localizer = localizer_class(self.target_size)

        recognizer_class = restore_backup(log_data['recognizer'], args.model_dir)
        log_keys = ['use_group_norm', 'num_layers', 'rotation_dropout_ratio', 'bos_token']
        recognizer_args = {key: log_data[key] for key in log_keys if key in log_data}

        self.recognizer = recognizer_class(
            self.data_loader.num_chars_per_word,
            self.data_loader.num_words_per_image,
            self.data_loader.num_classes,
            **recognizer_args,
        )

        if args.gpu is not None:
            self.localizer.to_gpu(args.gpu)
            if self.recognizer is not None:
                self.recognizer.to_gpu(args.gpu)

        self.evaluator = TextRecognitionTestFunction(
            self.localizer,
            self.recognizer,
            args.gpu,
            self.data_loader.blank_label,
            self.data_loader.char_map,
            return_best_result=args.return_only_best_result,
            strip_non_alpha_numeric_predictions=args.strip_non_alpha,
        )
        self.evaluator.xp = self.localizer.xp
        self.mean_calculator = TextRecognitionEvaluator(self.data_iterator, self.evaluator)

        self.bbox_plotter = TextRecognitionBBoxPlotter(
            self.data_loader.get_example(0)['image'][:, 0, ...],
            os.path.join(args.model_dir, 'eval_bboxes'),
            self.target_size,
            render_extracted_rois=False,
            device=args.gpu,
            num_rois_to_render=4,
            show_visual_backprop_overlay=False,
            show_backprop_and_feature_vis=False,
            visualization_anchors=[
                ["visual_backprop_anchors"],
            ],
            char_map=self.data_loader.char_map,
            blank_label=self.data_loader.blank_label,
            predictors={
                "localizer": self.localizer,
                "recognizer": self.recognizer,
            },
        )
        self.bbox_plotter.xp = self.localizer.xp

        self.results_path = os.path.join(self.args.model_dir, args.results_path)

    def load_weights(self, snapshot_name, model, dataset_name=''):
        with np.load(os.path.join(self.args.model_dir, snapshot_name)) as f:
            chainer.serializers.NpzDeserializer(f, strict=True).load(model)

        if self.args.save_predictions and hasattr(self, 'bbox_plotter'):
            new_out_dir = os.path.join(
                self.bbox_plotter.out_dir,
                dataset_name,
                os.path.splitext(snapshot_name)[0].split('_')[-1]
            )
            os.makedirs(new_out_dir, exist_ok=True)
            self.bbox_plotter.out_dir = new_out_dir

    def reset(self):
        self.data_iterator.reset()

    def evaluate(self, snapshot_name=''):
        reporter = reporter_module.Reporter()
        current_device = cuda.get_device_from_id(self.args.gpu)
        summary = reporter_module.DictSummary()
        with current_device, reporter, configuration.using_config('train', False):
            for i, batch in enumerate(tqdm(self.data_iterator, total=len(self.data_loader) // self.args.batchsize)):
                observation = {}
                batch = concat_examples(batch, self.args.gpu)
                image_size = Size._make(batch['image'].shape[-2:])

                with reporter_module.report_scope(observation):
                    rois, bboxes, text_predictions, best_indices, chosen_prediction, scores = self.evaluator(return_predictions=True, **batch)
                summary.add(observation)

                if self.args.save_predictions:
                    assert self.args.batchsize == 1, "if you want to save predictions, batchsize must be 1!"
                    batch_size, num_predictions, num_bboxes, num_channels, height, width = rois.shape

                    base_image = self.bbox_plotter.array_to_image(batch['image'][0, 0])
                    chosen_word = self.data_loader.decode_chars(cuda.to_cpu(text_predictions[0, chosen_prediction[0]].squeeze()))
                    base_image = self.bbox_plotter.render_text(base_image, base_image, chosen_word, 0, bottom=True)
                    rendered_images = [base_image]

                    iterator = zip(
                        F.separate(self.localizer.xp.stack(
                            [batch['image'][i, best_indices[i]] for i in range(self.args.batchsize)]), axis=1),
                        F.separate(rois, axis=1), F.separate(bboxes, axis=1), F.separate(text_predictions, axis=1),
                        F.separate(scores, axis=1)
                    )
                    for image, roi, bbox, prediction, score in iterator:
                        image = image.array
                        roi = roi.array
                        bbox = bbox.array
                        prediction = prediction.array
                        score = score.array

                        bbox = self.localizer.xp.reshape(bbox, (-1, 2, height, width))

                        predicted_words = self.data_loader.decode_chars(cuda.to_cpu(prediction.squeeze()))
                        predicted_words = f"{predicted_words} {format(float(score[0]), '.4f')}"

                        if args.cut_bboxes:
                            cut_length = batch['num_words'][0] if 'num_words' in batch else len(predicted_words)
                            bbox = bbox[:cut_length, ...]
                            roi = roi[:cut_length, ...]
                        if args.render_no_boxes:
                            bbox = bbox[:1]
                            roi = roi[:1]

                        rendered_images.append(
                            self.render_roi(
                                [],
                                bbox,
                                None,
                                i,
                                image,
                                roi,
                                predicted_words
                            )
                        )
                    self.save_rois(rendered_images, i)

            self.save_eval_results(snapshot_name, summary)

    def render_roi(self, backprop_visualizations, bboxes, class_predictions, index, image, rois, predicted_word):
        dest_image = self.bbox_plotter.render_rois(
            rois,
            bboxes.copy(),
            index,
            image[0],
            backprop_vis=backprop_visualizations,
        )
        if class_predictions is not None:
            dest_image = self.bbox_plotter.render_discriminator_result(
                dest_image,
                self.bbox_plotter.array_to_image(image[0].copy()),
                self.bbox_plotter.get_discriminator_output_function(chainer.Variable(class_predictions.transpose()))
            )

        if predicted_word is not None:
            dest_image = self.bbox_plotter.render_text(dest_image, dest_image, predicted_word, 0, bottom=True)

        return dest_image

    def save_rois(self, images, index):
        dest_image_width = max(images, key=lambda x: x.width).width
        dest_image_height = sum([image.height for image in images])
        dest_image = Image.new("RGB", (dest_image_width, dest_image_height))
        current_start_height = 0
        for image in images:
            dest_image.paste(image, (0, current_start_height))
            current_start_height += image.height

        self.bbox_plotter.save_image(dest_image, index)

    def save_eval_results(self, snapshot_name, summary):
        # calculate map for our detection
        eval_result = self.mean_calculator.calculate_mean_of_summary(summary)
        eval_result["snapshot_name"] = snapshot_name

        if os.path.exists(self.results_path):
            with open(self.results_path) as eval_file:
                json_data = json.load(eval_file)
        else:
            json_data = []

        json_data.append(eval_result)

        with open(self.results_path, 'w') as eval_file:
            json.dump(json_data, eval_file, indent=4)


def plot_eval_results(data, model_dir, dataset_name):
    values_per_key = defaultdict(list)

    for element in data:
        for key, value in element.items():
            values_per_key[key] += [value]

    for (key, value), color in zip(values_per_key.items(), get_next_color()):
        if key == 'snapshot_name':
            continue
        plt.plot(value, label=key)

    plt.legend()
    plt.savefig(os.path.join(model_dir, f"plot_{dataset_name}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluates trained localizer")
    parser.add_argument("eval_gt", help="path to gt file with all images to test")
    parser.add_argument("model_dir", help="path to directory containing train results")
    parser.add_argument("snapshot_prefix", help="prefix of snapshots to evaluate")
    parser.add_argument("--log-name", default="log", help="name of the log file [default: log]")
    parser.add_argument("--gpu", "-g", type=int, help="gpu to use [default: use cpu]")
    parser.add_argument("--num-samples", "-n", type=int, help="max number of samples to test [default: test all]")
    parser.add_argument("--batchsize", "-b", type=int, default=1, help="number of images to evaluate at once [default: 1]")
    parser.add_argument("--save-predictions", action='store_true', default=False, help="use bbox plotter to store the predicted bboxes for every test sample")
    parser.add_argument("--recognizer-name", required=True, help="name of recognizer to use")
    parser.add_argument("--force-reset", action='store_true', default=False, help="force a reset of eval results file")
    parser.add_argument("--char-map", help="path to a char map [default: get char map path from log file (does only work if evaluating on the same pc as training happened)]")
    parser.add_argument("--results-path", default="eval_results.json", help="path to file where eval results shall be saved")
    parser.add_argument("--do-not-cut-bboxes", dest="cut_bboxes", action='store_false', default=True, help="show all bboxes in plotted images")
    parser.add_argument("--render-no-boxes", action='store_true', default=False, help="indicate that no bboxes shall be shown while saving predictions")
    parser.add_argument("--dataset-name", "--dn", default='', help="name of dataset for saved predictions (makes it easier to have multiple evaluated datasets per model)")
    parser.add_argument("--render-all-results", action='store_false', dest='return_only_best_result', default=True, help="show the result of all three images")
    parser.add_argument("--strip-non-alpha", action='store_true', default=False, help="Strip all predicted non alpha numeric characters, as the dataset does not include any of those anyway")

    args = parser.parse_args()

    evaluator = Evaluator(args)
    if os.path.exists(evaluator.results_path) and not args.save_predictions:
        if args.force_reset:
            os.unlink(evaluator.results_path)
            evaluated_snapshots = []
        else:
            # we already evaluated some snapshots, so we do not need to do that again
            with open(evaluator.results_path) as already_evaluated_model_results:
                json_data = json.load(already_evaluated_model_results)
                evaluated_snapshots = [item['snapshot_name'] for item in json_data]
    else:
        evaluated_snapshots = []

    localizer_snapshots = list(
        sorted(
            filter(lambda x: x not in evaluated_snapshots and args.snapshot_prefix in x, os.listdir(args.model_dir)),
            key=lambda x: int(getattr(re.search(r"(\d+).npz", x), 'group', lambda: 0)(1))
        )
    )
    recognizer_snapshots = list(
        sorted(
            filter(lambda x: x not in evaluated_snapshots and args.recognizer_name in x, os.listdir(args.model_dir)),
            key=lambda x: int(getattr(re.search(r"(\d+).npz", x), 'group', lambda: 0)(1))
        )
    )
    for localizer_snapshot, recognizer_snapshot in tqdm(zip(localizer_snapshots, recognizer_snapshots),
                                                        total=len(localizer_snapshots)):

        evaluator.load_weights(localizer_snapshot, evaluator.localizer, args.dataset_name)
        evaluator.load_weights(recognizer_snapshot, evaluator.recognizer, args.dataset_name)
        evaluator.reset()
        evaluator.evaluate(localizer_snapshot)

    if os.path.exists(evaluator.results_path):
        with open(evaluator.results_path) as evaluated_model_results:
            json_data = json.load(evaluated_model_results)
            plot_eval_results(json_data, args.model_dir, args.dataset_name)
