import chainer.functions as F
import chainer.links as L

from text.lstm_text_localizer import LSTMTextLocalizer
from text.transformer_recognizer import TransformerTextRecognizer


class FSNSTransformerRecognizer(TransformerTextRecognizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with self.init_scope():
            self.feature_downscale = L.Linear(None, self.transformer_size)

    def extract_features(self, rois):
        features = super().extract_features(rois)

        batch_size, num_boxes, num_features = features.shape
        features = F.reshape(features, (batch_size // 4, 4, num_boxes, num_features))
        features = F.transpose(features, (0, 2, 3, 1))
        features = F.reshape(features, (batch_size // 4, num_boxes, num_features * 4))

        features = self.feature_downscale(features, n_batch_axes=2)

        return features


class FSNSLSTMLocalizer(LSTMTextLocalizer):

    def split_views(self, images):
        # reshape images in such a way that each view of FSNS is passed independently through network
        batch_size, num_channels, in_height, in_width = images.shape
        images = F.reshape(images, (batch_size, num_channels, in_height, 4, in_width // 4))
        images = F.transpose(images, (0, 3, 1, 2, 4))
        images = F.reshape(images, (batch_size * 4, num_channels, in_height, in_width // 4))
        return images

    def __call__(self, images):
        images = self.split_views(images)
        return super().__call__(images)
