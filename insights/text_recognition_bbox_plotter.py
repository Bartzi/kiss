import chainer.functions as F

from insights.bbox_plotter import BBOXPlotter


class TextRecognitionBBoxPlotter(BBOXPlotter):

    def __init__(self, *args, **kwargs):
        self.char_map = kwargs.pop('char_map')
        self.blank_label = kwargs.pop('blank_label')
        super().__init__(*args, **kwargs)

    def init_predictors(self, predictors):
        self.localizer = predictors['localizer']
        self.recognizer = predictors['recognizer']

    def decode_text_prediction(self, text_prediction):
        words = []
        for text_image in text_prediction:
            for word in text_image:
                word = ''.join(
                    [self.char_map[str(prediction.array)] for prediction in word if str(prediction.array) != self.blank_label]
                )
                words.append(word)
        return words

    def get_predictions(self, image):
        rois, bboxes, localizer_visual_backprop = self.localizer.predict(
            image[self.xp.newaxis, ...],
            return_visual_backprop=True
        )
        batch_size, num_rois, num_channels, height, width = rois.shape
        reshaped_rois = F.reshape(rois, (-1, num_channels, height, width))
        reshaped_bboxes = F.reshape(bboxes, (-1, 2, height, width))
        text_prediction = self.recognizer.predict(rois)
        # assessor_prediction, assessor_visual_backprop = self.assessor.predict(reshaped_rois, return_visual_backprop=True)

        return {
            "rois": reshaped_rois,
            "bboxes": reshaped_bboxes,
            "words": self.decode_text_prediction(text_prediction),
            # "assessor_prediction": assessor_prediction,
            "visual_backprop": {
                "localizer": getattr(localizer_visual_backprop, 'array', localizer_visual_backprop),
                # "assessor": getattr(assessor_visual_backprop, 'array', assessor_visual_backprop),
            }
        }

    def render_predictions(self, dest_image, predictions):
        # dest_image = self.render_discriminator_result(
        #     dest_image,
        #     self.array_to_image(self.image.copy()),
        #     self.get_discriminator_output_function(predictions["assessor_prediction"])
        # )

        text_line = ' '.join(
            [''.join(
                char for char in word if char != self.char_map[str(self.blank_label)]
            ) for word in predictions["words"]]
        )
        dest_image = self.render_text(dest_image, dest_image, text_line, 0, bottom=True)
        return dest_image
