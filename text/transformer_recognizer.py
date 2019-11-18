import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain

from resnet.resnet_gn import ResNet
from train_utils.autocopy import maybe_copy
from transformer import get_conv_feature_encoder_decoder
from transformer.utils import subsequent_mask


class TransformerTextRecognizer(Chain):

    def __init__(self, num_chars, num_words, num_classes, bos_token, **kwargs):
        super().__init__()
        self.transformer_size = kwargs.pop('transformer_size', 512)
        self.bos_token = bos_token

        with self.init_scope():
            self.feature_extractor = ResNet(kwargs.pop('num_layers', 18))
            self.transformer = get_conv_feature_encoder_decoder(num_classes, N=1, model_size=self.transformer_size)
            self.classifier = L.Linear(self.transformer_size, num_classes)
            self.mask = subsequent_mask(self.transformer_size)

        self.num_chars = num_chars
        self.num_words = num_words
        chainer.global_config.user_text_recognition_grayscale_input = False

    @maybe_copy
    def __call__(self, rois, labels):
        h = self.extract_features(rois)

        labels = self.adapt_labels(labels, len(rois))
        transformer_output = self.transformer(h, labels, None, self.mask)

        # return shape: batch_size, num_bboxes, num_classes
        return self.classifier(transformer_output, n_batch_axes=2)

    def get_bos_token_array(self, batch_size, num_words):
        bos_token = self.xp.full((batch_size, num_words, 1), self.bos_token, dtype=self.xp.int32)
        return bos_token

    def adapt_labels(self, labels, batch_size):
        batch_size, num_words, num_chars = labels.shape
        bos_token = self.get_bos_token_array(batch_size, num_words)

        # shift labels to the right by adding the bos token
        shifted_labels = self.xp.concatenate([bos_token, labels[:, :, :-1]], axis=2)
        # remove extra dim and add it as batch size
        return self.xp.reshape(shifted_labels, (-1, num_chars))

    def extract_features(self, rois):
        batch_size, num_bboxes, num_channels, height, width = rois.shape
        rois = F.reshape(rois, (-1, num_channels, height, width))

        h = self.feature_extractor(rois)
        intermediate_batch_size, num_channels, feature_height, feature_width = h.shape
        h = F.average_pooling_2d(h, (feature_height, feature_width))
        # h = F.reshape(h, (intermediate_batch_size, num_channels, -1))

        return F.reshape(h, (batch_size, num_bboxes, -1))

    def calc_loss(self, predictions, labels):
        labels = labels.ravel()
        predictions = F.reshape(predictions, (-1, predictions.shape[-1]))
        return F.softmax_cross_entropy(predictions, labels)

    def decode_prediction(self, prediction):
        return F.argmax(F.softmax(prediction, axis=2), axis=2)

    @maybe_copy
    def predict(self, images, return_raw_classification_result=False):
        feature_map = self.extract_features(images)
        memory = self.transformer.encode(feature_map, None)

        target = self.get_bos_token_array(len(images), self.num_words)
        target = self.xp.reshape(target, (-1, 1))
        char = None

        for _ in range(self.num_chars):
            decoded = self.transformer.decode(memory, None, target, self.mask)
            char = self.classifier(decoded, n_batch_axes=2)
            predicted_chars = self.decode_prediction(char)
            target = F.concat([target, predicted_chars[:, -1:]])

        result = F.expand_dims(target[:, 1:], 1)
        if return_raw_classification_result:
            return result, char
        return result
