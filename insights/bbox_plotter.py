import base64
import json
import os
import socket
from io import BytesIO

import chainer.functions as F
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont

from PIL import ImageDraw

import chainer
from chainer.backends import cuda
from chainer.training import Extension

from common.utils import Size
from insights.visual_backprop import VisualBackprop


COLOR_MAP = [
    "#00B3FF",  # Vivid Yellow
    "#753E80",  # Strong Purple
    "#0068FF",  # Vivid Orange
    "#D7BDA6",  # Very Light Blue
    "#2000C1",  # Vivid Red
    "#62A2CE",  # Grayish Yellow
    "#667081",  # Medium Gray

    # The following don't work well for people with defective color vision
    "#347D00",  # Vivid Green
    "#8E76F6",  # Strong Purplish Pink
    "#8A5300",  # Strong Blue
    "#5C7AFF",  # Strong Yellowish Pink
    "#7A3753",  # Strong Violet
    "#008EFF",  # Vivid Orange Yellow
    "#5128B3",  # Strong Purplish Red
    "#00C8F4",  # Vivid Greenish Yellow
    "#0D187F",  # Strong Reddish Brown
    "#00AA93",  # Vivid Yellowish Green
    "#153359",  # Deep Yellowish Brown
    "#133AF1",  # Vivid Reddish Orange
    "#162C23",  # Dark Olive Green
]


def get_next_color():
    while True:
        for color in COLOR_MAP:
            yield color


class BBOXPlotter(Extension):

    def __init__(self, image, out_dir, out_size, **kwargs):
        super(BBOXPlotter, self).__init__()
        self.image = image
        self.reference_image = kwargs.pop("reference_image", None)
        self.reference_features = None
        self.render_extracted_rois = kwargs.pop("render_extracted_rois", True)
        self.image_size = Size(height=image.shape[1], width=image.shape[2])
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_size = out_size
        self.colours = get_next_color
        self.send_bboxes = kwargs.pop("send_bboxes", False)
        self.upstream_ip = kwargs.pop("upstream_ip", '127.0.0.1')
        self.upstream_port = kwargs.pop("upstream_port", 1337)
        self.font = ImageFont.truetype("train_utils/DejaVuSans.ttf", 14)
        self.visualization_anchors = kwargs.pop("visualization_anchors", [])
        self.visual_backprop = VisualBackprop()
        self.vis_features = kwargs.pop("feature_anchors", [])
        self.plot_objectness_classification_result = kwargs.pop('plot_objectness_classification_result', False)
        self.show_visual_backprop_overlay = kwargs.pop('show_visual_backprop_overlay', False)
        # index of the visual backrpop prediction that is to be shown in overlay
        self.visual_backprop_index = kwargs.pop('visual_backprop_index', 0)
        self.show_backprop_and_feature_vis = kwargs.pop('show_backprop_and_feature_vis', False)
        self.get_discriminator_output_function = kwargs.pop('discriminator_output_function', self.get_discriminator_output)
        self.render_pca = kwargs.pop('render_pca', False)
        self.gt_bbox = kwargs.pop('gt_bbox', None)
        self.xp = np
        self.devices = kwargs.pop('devices', None)
        self.log_name = kwargs.pop('log_name', 'training')
        self.max_num_rois_to_render = kwargs.pop('num_rois_to_render', None)
        self.sort_rois = kwargs.pop('sort_rois', False)

        self.init_predictors(kwargs.pop("predictors", {}))

    def init_predictors(self, predictors):
        self.localizer = predictors['localizer']
        # self.assessor = predictors['assessor']

    def initialize(self, trainer):
        # run the network with the completely randomized state we start with
        self(trainer)

    def send_image(self, data):
        height = data.height
        width = data.width
        channels = len(data.getbands())

        # convert image to png in order to save network bandwidth
        png_stream = BytesIO()
        data.save(png_stream, format="PNG")
        png_stream = png_stream.getvalue()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.upstream_ip, self.upstream_port))
            except Exception as e:
                print(e)
                print("could not connect to display server, disabling image rendering")
                self.send_bboxes = False
                return
            data = {
                'width': width,
                'height': height,
                'channels': channels,
                'title': self.log_name,
                'image': base64.b64encode(png_stream).decode('utf-8'),
            }
            sock.send(bytes(json.dumps(data), 'utf-8'))

    def array_to_image(self, array):
        if array.shape[0] == 1:
            # image is black and white, we need to trick the system into thinking, that we are having an RGB image
            array = self.xp.tile(array, (3, 1, 1))
        array = array.copy() * 255
        return Image.fromarray(cuda.to_cpu(array.transpose(1, 2, 0)).astype(np.uint8), "RGB").convert("RGBA")

    def variable_to_image(self, data):
        if isinstance(data, chainer.Variable):
            data = data.data
        return self.array_to_image(data)

    def get_predictions(self, image):
        rois, bboxes, localizer_visual_backprop = self.localizer.predict(
            image[self.xp.newaxis, ...],
            return_visual_backprop=True
        )
        # assessor_prediction, assessor_visual_backprop = self.assessor.predict(rois, return_visual_backprop=True)

        return {
            "rois": rois,
            "bboxes": bboxes,
            # "assessor_prediction": assessor_prediction,
            "visual_backprop": {
                "localizer": getattr(localizer_visual_backprop, 'array', localizer_visual_backprop),
                # "assessor": getattr(assessor_visual_backprop, 'array', assessor_visual_backprop),
            }
        }

    def sort_predictions(self, predictions):
        rois, assessor_output, assessor_visual_backprop = self.sort_rois_and_scores(
            predictions["rois"],
            predictions["assessor_prediction"],
            self.assessor,
            roi_visual_backprop=predictions["visual_backprop"]["assessor"]
        )
        predictions["rois"] = rois
        predictions["assessor_prediction"] = assessor_output
        predictions["visual_backprop"]["assessor"] = assessor_visual_backprop
        return predictions

    def filter_predictions(self, predictions):
        predictions["rois"] = predictions["rois"][:self.max_num_rois_to_render]
        # predictions["assessor_prediction"] = predictions["assessor_prediction"][:self.max_num_rois_to_render]
        # predictions["visual_backprop"]["assessor"] = predictions["visual_backprop"]["assessor"][:self.max_num_rois_to_render]
        return predictions

    def render_predictions(self, dest_image, predictions):
        return self.render_discriminator_result(
            dest_image,
            self.array_to_image(self.image.copy()),
            self.get_discriminator_output_function(predictions["assessor_prediction"])
        )

    def __call__(self, trainer):
        iteration = trainer.updater.iteration

        with chainer.using_device(trainer.updater.get_optimizer('opt_gen').target.device), chainer.using_config('train', False):
            self.xp = trainer.updater.get_optimizer('opt_gen').target.device.xp
            image = self.xp.asarray(self.image)
            predictions = self.get_predictions(image)

            if self.sort_rois:
                predictions = self.sort_predictions(predictions)

            if self.render_extracted_rois and self.max_num_rois_to_render is not None:
                predictions = self.filter_predictions(predictions)

            dest_image = self.render_rois(
                predictions["rois"],
                predictions["bboxes"],
                iteration,
                self.image.copy(),
                backprop_vis=predictions["visual_backprop"]['localizer'],
            )

            # self.show_backprop_vis(
            #     predictions["visual_backprop"]["assessor"],
            #     dest_image,
            #     self.array_to_image(self.image.copy()),
            #     count=1
            # )

            dest_image = self.render_predictions(dest_image, predictions)

            if self.gt_bbox is not None:
                dest_image = self.draw_gt_bbox(dest_image)
            if self.render_pca and self.render_extracted_rois:
                dest_image = self.show_pca(dest_image, trainer.updater)
            self.save_image(dest_image, iteration)

    def get_feature_maps(self, predictor):
        feature_visualizations = []
        for feature_anchor in self.vis_features:
            targets = predictor
            for attr in feature_anchor:
                targets = getattr(targets, attr, None)
            if targets is not None:
                for target in targets:
                    feature_visualizations.append(self.show_feature_map(target))
        return feature_visualizations

    def get_backprop_visualization(self, predictor):
        backprop_visualizations = []
        for visanchor in self.visualization_anchors:
            vis_targets = predictor
            for target in visanchor:
                vis_targets = getattr(vis_targets, target)
            if vis_targets is not None:
                if not hasattr(vis_targets, '__iter__'):
                    vis_targets = [vis_targets]
                for vis_target in vis_targets:
                    backprop_visualizations.append(self.visual_backprop.perform_visual_backprop(vis_target))
        return backprop_visualizations

    @property
    def original_image_paste_location(self):
        return 0, 0

    def compose_image_and_visual_backprop(self, original_image, backprop_image):
        backprop_image = self.array_to_image(
            self.xp.tile(backprop_image, (3, 1, 1))
        ).resize(
            (self.image_size.width, self.image_size.height)
        )
        original_image = original_image.convert("RGBA")
        backprop_image = backprop_image.convert("RGBA")

        resulting_image = Image.blend(original_image, backprop_image, 0.6)
        return resulting_image

    def render_rois(self, rois, bboxes, iteration, image, backprop_vis=(), feature_vis=()):
        image = self.array_to_image(image)

        num_timesteps = self.get_num_timesteps(bboxes)
        bboxes, dest_image = self.set_output_sizes(backprop_vis, feature_vis, bboxes, image, num_timesteps, len(rois))

        if self.render_extracted_rois:
            self.render_extracted_regions(dest_image, image, rois)

        if len(backprop_vis) != 0 and self.show_backprop_and_feature_vis:
            # if we have a backprop visualization we can show it now
            self.show_backprop_vis(backprop_vis, dest_image, image)

        if self.show_visual_backprop_overlay and len(backprop_vis) != 0:
            backprop_image_to_show = backprop_vis[self.visual_backprop_index][0]
            image = self.compose_image_and_visual_backprop(image, backprop_image_to_show)

        if len(feature_vis) != 0 and self.show_backprop_and_feature_vis:
            self.show_backprop_vis(feature_vis, dest_image, image, image.height)

        self.draw_bboxes(bboxes, image)
        dest_image.paste(image, self.original_image_paste_location)
        return dest_image

    def sort_rois_and_scores(self, rois, scores, assessor, roi_visual_backprop=None):
        # sort rois based on scores obtained by assessor, but only sort using first prediction
        sort_scores = assessor.extract_iou_prediction(scores)
        sort_scores = sort_scores.data.copy()
        score_indices = sort_scores.argsort()[::-1]
        rois = rois[score_indices]
        scores = scores[score_indices]
        if roi_visual_backprop is not None:
            roi_visual_backprop = roi_visual_backprop[score_indices]
            return rois, scores, roi_visual_backprop
        return rois, scores

    def save_image(self, dest_image, iteration):
        dest_image.save("{}.png".format(os.path.join(self.out_dir, str(iteration))), 'png')
        if self.send_bboxes:
            self.send_image(dest_image)

    def get_num_timesteps(self, bboxes):
        return bboxes.shape[0]

    def set_output_sizes(self, backprop_vis, feature_vis, bboxes, image, num_timesteps, num_rois_to_render):
        _, num_channels, height, width = bboxes.shape

        image_height = image.height if (len(backprop_vis) == 0 or not self.show_backprop_and_feature_vis) and not self.render_pca else image.height + self.image_size.height
        image_height = image_height + self.image_size.height if len(feature_vis) > 0 and self.show_backprop_and_feature_vis else image_height
        image_width = image.width + image.width * num_rois_to_render if self.render_extracted_rois else image.width

        dest_image = Image.new("RGBA", (image_width, image_height), color='black')
        bboxes = F.reshape(bboxes, (num_timesteps, 1, num_channels, height, width))

        return bboxes, dest_image

    def show_backprop_vis(self, visualizations, dest_image, image, height_offset=0, count=0):
        for visualization in visualizations:
            for vis in visualization:
                backprop_image = self.array_to_image(self.xp.tile(vis, (3, 1, 1))).resize(
                    (self.image_size.width, self.image_size.height))
                dest_image.paste(backprop_image, (count * backprop_image.width, height_offset + image.height))
                count += 1

    def show_feature_map(self, feature_map):
        with chainer.no_backprop_mode():
            averaged_feature_map = F.average(feature_map, axis=1, keepdims=True)[0]
            averaged_feature_map -= averaged_feature_map.data.min()
            max_value = averaged_feature_map.data.max()
            if max_value > 0:
                averaged_feature_map /= max_value
        return averaged_feature_map[None, ...].data

    def show_pca(self, dest_image, updater):
        colors = ['navy', 'turquoise', 'darkorange']
        if getattr(updater, 'pca', None) is None:
            return dest_image
        pca_discriminator = updater.pca.reshape(3, -1, updater.n_components_pca)

        plt.figure()
        for i, color, in enumerate(colors):
            plt.scatter(pca_discriminator[i, :, 0], pca_discriminator[i, :, 1], color=color, lw=2)
        plt.legend(['fake', 'real', 'anchor'])

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        image = image.resize((self.image_size.width, self.image_size.height), Image.LANCZOS)
        dest_image.paste(image, (self.image_size.width, self.image_size.height))
        plt.close()
        return dest_image

    def render_extracted_regions(self, dest_image, image, rois):
        num_rois, num_channels, height, width = rois.shape
        if num_rois == 0:
            return
        rois = rois.reshape(len(rois), -1, num_channels, height, width)

        for i, roi in enumerate(rois, start=1):
            roi_image = self.variable_to_image(roi[0])
            paste_location = i * image.width, 0
            dest_image.paste(roi_image.resize((self.image_size.width, self.image_size.height)), paste_location)

    def draw_bboxes(self, bboxes, image):
        if len(bboxes) == 0:
            return
        draw = ImageDraw.Draw(image)
        for i, sub_box in enumerate(F.separate(bboxes, axis=1)):
            for bbox, colour in zip(F.separate(sub_box, axis=0), self.colours()):
                bbox.data[...] = (bbox.data[...] + 1) / 2
                bbox.data[0, :] *= self.image_size.width
                bbox.data[1, :] *= self.image_size.height

                x = self.xp.clip(bbox.data[0, :].reshape(self.out_size), 0, self.image_size.width) + i * self.image_size.width
                y = self.xp.clip(bbox.data[1, :].reshape(self.out_size), 0, self.image_size.height)

                top_left = (x[0, 0], y[0, 0])
                top_right = (x[0, -1], y[0, -1])
                bottom_left = (x[-1, 0], y[-1, 0])
                bottom_right = (x[-1, -1], y[-1, -1])

                corners = [top_left, top_right, bottom_right, bottom_left]
                self.draw_bbox(colour, corners, draw)

    def draw_bbox(self, colour, corners, draw):
        next_corners = corners[1:] + [corners[0]]
        for first_corner, next_corner in zip(corners, next_corners):
            draw.line([first_corner, next_corner], fill=colour, width=3)

    def get_discriminator_output(self, discriminator_result):
        if discriminator_result.shape[1] > 1:
            discriminator_result = F.softmax(discriminator_result, axis=1)
        results = []
        for result in discriminator_result:
            if result.shape[0] == 1:
                result = format(float(result.data), ".3f")
            else:
                result = str(int(result.data.argmax()))
            results.append(result)
        return results

    def render_discriminator_result(self, dest_image, source_image, discriminator_result):
        for i, result in enumerate(discriminator_result, start=1):
            dest_image = self.render_text(dest_image, source_image, result, i)
        return dest_image

    def render_text(self, dest_image, source_image, text, i, bottom=False):
        label_image = Image.new(dest_image.mode, dest_image.size)
        draw = ImageDraw.Draw(label_image)
        paste_width = (i + 1) * source_image.width
        text_width, text_height = draw.textsize(text, self.font)
        insert_height = source_image.height - text_height - 1 if bottom else 0
        draw.rectangle([paste_width - text_width - 1, insert_height, paste_width, insert_height + text_height],
                       fill=(255, 255, 255, 160))
        draw.text((paste_width - text_width - 1, insert_height), text, fill='green', font=self.font)
        dest_image = Image.alpha_composite(dest_image, label_image)
        return dest_image

    def draw_gt_bbox(self, image):
        draw = ImageDraw.Draw(image)
        for bbox in self.gt_bbox:
            top_left = bbox[1], bbox[0]
            top_right = bbox[3], bbox[0]
            bottom_left = bbox[1], bbox[2]
            bottom_right = bbox[3], bbox[2]

            colour = COLOR_MAP[-1]
            self.draw_bbox(colour, [top_left, top_right, bottom_right, bottom_left], draw)
        return image


class ObjectnessBBOXPlotter(BBOXPlotter):

    def __init__(self, *args, **kwargs):
        self.tensorboard_writer = kwargs.pop('tensorboard_writer', None)
        super().__init__(*args, **kwargs)

    def __call__(self, trainer):
        iteration = trainer.updater.iteration

        if self.devices is not None:
            devices = self.devices
        else:
            devices = [cuda.get_device_from_id(trainer.updater.get_optimizer('opt_gen').target._device_id) for _ in range(2)]

        with chainer.using_config('train', False), cuda.Device(devices[0]):
            self.xp = np if trainer.updater.get_optimizer('opt_gen').target._device_id < 0 else cuda.cupy
            image = self.xp.array(self.image)
            predictor = trainer.updater.get_optimizer('opt_gen').target
            rois, bboxes, objectness_scores = predictor(image[self.xp.newaxis, ...])[:3]

            if len(rois.shape) > 4:
                rois = F.reshape(rois, (-1,) + rois.shape[-3:])
                bboxes = F.reshape(bboxes, (-1,) + bboxes.shape[-3:])
                objectness_scores = F.reshape(objectness_scores, (-1, objectness_scores.shape[-1]))

            discriminator = trainer.updater.get_optimizer('opt_dis').target
            class_predictions = discriminator(rois)

            backprop_visualizations = self.get_backprop_visualization(predictor)
            feature_visualizations = self.get_feature_maps(predictor)

            self.render_rois(
                rois,
                bboxes,
                iteration,
                self.image.copy(),
                backprop_vis=backprop_visualizations,
                feature_vis=feature_visualizations,
                objectness_scores=objectness_scores,
                class_predictions=class_predictions,
            )

    def render_rois(self, rois, bboxes, iteration, image, backprop_vis=(), feature_vis=(), objectness_scores=None, class_predictions=None):
        # get the predicted text
        image = self.array_to_image(image)

        num_timesteps = self.get_num_timesteps(bboxes)
        bboxes, dest_image = self.set_output_sizes(backprop_vis, feature_vis, bboxes, image, num_timesteps)
        if self.render_extracted_rois:
            self.render_extracted_regions(dest_image, image, rois, num_timesteps)

        if len(backprop_vis) != 0 and self.show_backprop_and_feature_vis:
            # if we have a backprop visualization we can show it now
            self.show_backprop_vis(backprop_vis, dest_image, image)

        if self.show_visual_backprop_overlay and len(backprop_vis) != 0:
            image = self.compose_image_and_visual_backprop(image, backprop_vis[0][0])

        if len(feature_vis) != 0 and self.show_backprop_and_feature_vis:
            self.show_backprop_vis(feature_vis, dest_image, image, image.height)

        self.draw_bboxes(bboxes, image)
        dest_image.paste(image, self.original_image_paste_location)

        if objectness_scores is not None and self.render_extracted_rois:
            dest_image = self.render_objectness_scores(objectness_scores, dest_image, image)

        if class_predictions is not None and self.render_extracted_rois:
            dest_image = self.render_objectness_scores(class_predictions, dest_image, image, bottom=True)

        dest_image.save("{}.png".format(os.path.join(self.out_dir, str(iteration))), 'png')

        if self.tensorboard_writer is not None:
            # show image in tensorboard
            self.tensorboard_writer.add_image("bbox_vis", np.asarray(dest_image).transpose(2, 0, 1).astype('float32') / 255, iteration)

        if self.send_bboxes:
            self.send_image(dest_image)

    def render_objectness_scores(self, objectness_scores, dest_image, image, bottom=False):
        if len(objectness_scores) == 0:
            return dest_image

        if isinstance(objectness_scores, tuple) and self.plot_objectness_classification_result:
            objectness_scores = objectness_scores[1]
            objectness_scores = F.softmax(objectness_scores)
            objectness_classification = map(lambda x: "{:.2f}".format(float(x)), objectness_scores[:, 1].data)
        elif isinstance(objectness_scores, tuple) and not self.plot_objectness_classification_result:
            objectness_scores = objectness_scores[0]
            objectness_classification = map(lambda x: "{:.2f}".format(float(F.mean(x).data)), objectness_scores[:, 0].data)
        elif objectness_scores.shape[-1] == 2:
            objectness_scores = F.softmax(objectness_scores)
            objectness_classification = map(lambda x: "{:.2f}".format(float(x)), objectness_scores[:, 1].data)
        else:
            objectness_classification = map(lambda x: "{:.2f}".format(float(F.mean(x).data)), objectness_scores[:, 0].data)

        for i, char in enumerate(objectness_classification, start=1):
            label_image = Image.new(dest_image.mode, dest_image.size)
            draw = ImageDraw.Draw(label_image)
            paste_width = (i + 1) * image.width
            text_width, text_height = draw.textsize(char, self.font)
            insert_height = image.height - text_height - 1 if bottom else 0
            draw.rectangle([paste_width - text_width - 1, insert_height, paste_width, insert_height + text_height], fill=(255, 255, 255, 160))
            draw.text((paste_width - text_width - 1, insert_height), char, fill='green', font=self.font)
            dest_image = Image.alpha_composite(dest_image, label_image)
        return dest_image

