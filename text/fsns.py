import chainer.functions as F
import chainer.links as L

from text.lstm_text_localizer import LSTMTextLocalizer
from text.transformer_recognizer import TransformerTextRecognizer
from text.transformer_text_localizer import TransformerTextLocalizer
from train_utils.autocopy import maybe_copy


def split_views(images):
    # reshape images in such a way that each view of FSNS is passed independently through network
    batch_size, num_channels, in_height, in_width = images.shape
    images = F.reshape(images, (batch_size, num_channels, in_height, 4, in_width // 4))
    images = F.transpose(images, (0, 3, 1, 2, 4))
    images = F.reshape(images, (batch_size * 4, num_channels, in_height, in_width // 4))
    return images


class FSNSTransformerRecognizer(TransformerTextRecognizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # with self.init_scope():
            # self.feature_downscale = L.Linear(None, self.transformer_size)

    def extract_features(self, rois):
        batch_size, num_bboxes, num_channels, height, width = rois.shape
        rois = F.reshape(rois, (-1, num_channels, height, width))

        h = self.feature_extractor(rois)
        intermediate_batch_size, num_channels, feature_height, feature_width = h.shape
        h = F.average_pooling_2d(h, (feature_height, feature_width))
        # h = F.reshape(h, (intermediate_batch_size, num_channels, -1))

        combined_batch_size, num_channels, height, width = h.shape
        h = F.reshape(h, (combined_batch_size // 4, 4 * num_channels, height, width))

        # h = F.reshape(h, (batch_size // 4, num_bboxes, -1))
        h = F.reshape(h, (batch_size // 4 * num_bboxes, 1, -1))
        # h = self.feature_downscale(h, n_batch_axes=2)

        return h

    @maybe_copy
    def predict(self, images, return_raw_classification_result=False):
        batch_size = len(images) // 4
        feature_map = self.extract_features(images)
        memory = self.transformer.encode(feature_map, None)

        target = self.get_bos_token_array(batch_size, self.num_words)
        target = self.xp.reshape(target, (-1, 1))

        char = None
        for _ in range(self.num_chars):
            decoded = self.transformer.decode(memory, None, target, self.mask)
            char = self.classifier(decoded, n_batch_axes=2)
            predicted_chars = self.decode_prediction(char)
            target = F.concat([target, predicted_chars[:, -1:]])

        result = F.reshape(target[:, 1:], (batch_size, self.num_words, self.num_chars))
        if return_raw_classification_result:
            return result, char
        return result


class FSNSLSTMLocalizer(LSTMTextLocalizer):

    def split_views(self, images):
        return split_views(images)

    def __call__(self, images):
        images = split_views(images)
        return super().__call__(images)


class FSNSTransformerLocalizer(TransformerTextLocalizer):

    def split_views(self, images):
        return split_views(images)

    def __call__(self, images):
        images = split_views(images)
        return super().__call__(images)
