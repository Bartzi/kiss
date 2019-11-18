import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer.backends import cuda

from resnet.resnet_gn import ResNet
from train_utils.autocopy import maybe_copy


class TextRecognizer(Chain):

    def __init__(self, num_chars, num_classes, **kwargs):
        super().__init__()

        with self.init_scope():
            self.feature_extractor = ResNet(kwargs.pop('num_layers', 18))
            # self.lstm = L.LSTM(None, 1024)
            self.classifier = L.Linear(None, num_classes)

        self.num_chars = num_chars
        chainer.global_config.user_text_recognition_grayscale_input = False

    @maybe_copy
    def __call__(self, rois):
        batch_size, num_bboxes, num_channels, height, width = rois.shape
        rois = F.reshape(rois, (-1, num_channels, height, width))

        # if not chainer.config.user_text_recognition_grayscale_input:
        #     # convert data to grayscale
        #     assert rois.shape[1] == 3, "rois are not in RGB, can not convert them to grayscale"
        #     r, g, b = F.separate(rois, axis=1)
        #     grey = 0.299 * r + 0.587 * g + 0.114 * b
        #     rois = F.stack([grey, grey, grey], axis=1)

        h = self.feature_extractor(rois)
        _, num_channels, feature_height, feature_width = h.shape
        h = F.average_pooling_2d(h, (feature_height, feature_width))

        h = F.reshape(h, (batch_size, num_bboxes, num_channels, -1))

        all_predictions = []
        for box in F.separate(h, axis=1):
            # box_predictions = [self.classifier(self.lstm(box)) for _ in range(self.num_chars)]
            box_predictions = [self.classifier(box) for _ in range(self.num_chars)]
            all_predictions.append(F.stack(box_predictions, axis=1))

        # return shape: batch_size, num_bboxes, num_chars, num_classes
        return F.stack(all_predictions, axis=2)

    def calc_loss(self, predictions, labels):
        recognition_losses = []
        assert predictions.shape[1] == labels.shape[1], "Number of boxes is not equal in predictions and labels"
        for box, box_labels in zip(F.separate(predictions, axis=1), F.separate(labels, axis=1)):
            assert box.shape[1] == box_labels.shape[1], "Number of predicted chars is not equal to number of chars in label"
            box_losses = [
                F.softmax_cross_entropy(char, char_label, reduce="no")
                for char, char_label in zip(F.separate(box, axis=1), F.separate(box_labels, axis=1))
            ]
            recognition_losses.append(F.stack(box_losses))
        return F.mean(F.stack(recognition_losses))

    def decode_prediction(self, prediction):
        words = []
        for box in F.separate(prediction, axis=1):
            word = [F.argmax(F.softmax(character), axis=1) for character in F.separate(box, axis=1)]
            words.append(F.stack(word, axis=1))

        return F.stack(words, axis=1)

    @maybe_copy
    def predict(self, images, return_visual_backprop=False):
        if isinstance(images, list):
            images = [self.xp.asarray(image) for image in images]
            images = self.xp.stack(images, axis=0)

        with chainer.using_config('train', False):
            text_recognition_result = self(images)
            prediction = self.decode_prediction(text_recognition_result)

        return prediction
