import chainer
import chainer.functions as F
from PIL import Image

from insights.text_recognition_bbox_plotter import TextRecognitionBBoxPlotter
from train_utils.datatypes import Size


class FSNSBBoxPlotter(TextRecognitionBBoxPlotter):

    def __init__(self, *args, **kwargs):
        kwargs['render_extracted_rois'] = False
        kwargs['sort_rois'] = False
        super().__init__(*args, **kwargs)

        self.image_size = Size(width=150, height=150)

    def __call__(self, trainer):
        iteration = trainer.updater.iteration

        with chainer.using_device(trainer.updater.get_optimizer('opt_gen').target.device), chainer.using_config('train', False):
            self.xp = trainer.updater.get_optimizer('opt_gen').target.device.xp
            image = self.xp.asarray(self.image)
            predictions = self.get_predictions(image)

            images = self.localizer.split_views(image[self.xp.newaxis, ...]).array
            rois = F.reshape(predictions['rois'], (len(images), -1) + predictions['rois'].shape[1:])
            bboxes = F.reshape(predictions['bboxes'], (len(images), -1) + predictions['bboxes'].shape[1:])
            dest_images = []
            for image, roi, bbox, visual_backprop in zip(images, rois, bboxes, predictions["visual_backprop"]['localizer']):
                dest_image = self.render_rois(
                    roi,
                    bbox,
                    iteration,
                    image,
                    backprop_vis=visual_backprop[self.xp.newaxis, ...],
                )

                if self.gt_bbox is not None:
                    dest_image = self.draw_gt_bbox(dest_image)
                if self.render_pca and self.render_extracted_rois:
                    dest_image = self.show_pca(dest_image, trainer.updater)
                dest_images.append(dest_image)

            dest_image = self.merge_images(dest_images)
            dest_image = self.render_predictions(dest_image, predictions)
            self.save_image(dest_image, iteration)

    def merge_images(self, images):
        overall_width = sum([image.width for image in images])
        overall_height = images[0].height

        dest_image = Image.new("RGBA", (overall_width, overall_height))
        current_width = 0
        for i, image in enumerate(images):
            dest_image.paste(image, (current_width, 0))
            current_width += image.width

        return dest_image
