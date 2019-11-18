import os

import numpy

from PIL import Image

from common.datasets.image_dataset import prepare_image
from common.datasets.text_recognition_image_dataset import TextRecognitionImageDataset


class TextRecognitionEvaluationDataset(TextRecognitionImageDataset):

    def load_image(self, file_name):
        with Image.open(os.path.join(self.root, file_name)) as the_image:
            the_image = the_image.convert(self.image_mode).convert("RGB")

            images = [the_image]
            if the_image.height > the_image.width * 1.3:
                clockwise_image = the_image.rotate(-90, Image.BICUBIC, True)
                counter_clockwise_image = the_image.rotate(90, Image.BICUBIC, True)
                images.extend([clockwise_image, counter_clockwise_image])
            else:
                images.extend([
                    the_image.rotate(-5, Image.BICUBIC, True),
                    the_image.rotate(5, Image.BICUBIC, True),
                ])

            images = [
                prepare_image(
                    image,
                    self.image_size,
                    numpy,
                    self.keep_aspect_ratio,
                    normalize=self.full_normalize,
                    do_resize=self.resize_after_load,
                ) for image in images
            ]
            image = numpy.stack(images, axis=0)
        return image
