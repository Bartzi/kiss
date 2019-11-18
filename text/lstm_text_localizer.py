import chainer.functions as F
import chainer.links as L
from chainer.links.model.vision.resnet import _global_average_pooling_2d

from functions.rotation_droput import rotation_dropout
from text.text_localizer import TextLocalizer


class LSTMTextLocalizer(TextLocalizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with self.init_scope():
            self.lstm = L.LSTM(None, 1024)

            self.param_predictor = L.Linear(1024, 6)
            # params = self.param_predictor.b.data
            # params[...] = 0
            # params[[0, 4]] = 0.8
            # self.param_predictor.W.data[...] = 0

    def __call__(self, images):
        self.lstm.reset_state()
        return super().__call__(images)

    def get_transform_params(self, features):
        h = _global_average_pooling_2d(features)
        lstm_predictions = [self.lstm(h) for _ in range(self.num_bboxes_to_localize)]
        lstm_predictions = F.stack(lstm_predictions, axis=1)
        batch_size, num_boxes, _ = lstm_predictions.shape
        lstm_predictions = F.reshape(lstm_predictions, (-1,) + lstm_predictions.shape[2:])

        params = self.param_predictor(lstm_predictions)
        transform_params = rotation_dropout(F.reshape(params, (-1, 2, 3)), ratio=self.dropout_ratio)
        return transform_params


class SliceLSTMLocalizer(LSTMTextLocalizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with self.init_scope():
            self.pre_transform_params = L.Linear(None, self.num_bboxes_to_localize * self.features_per_timestep)

    def get_transform_params(self, features):
        h = self.pre_transform_params(features)
        slices = F.split_axis(h, self.num_bboxes_to_localize, axis=1)

        lstm_predictions = [self.lstm(slice) for slice in slices]
        lstm_predictions = F.stack(lstm_predictions, axis=1)
        batch_size, num_boxes, _ = lstm_predictions.shape
        lstm_predictions = F.reshape(lstm_predictions, (-1,) + lstm_predictions.shape[2:])

        params = self.param_predictor(lstm_predictions)
        transform_params = rotation_dropout(F.reshape(params, (-1, 2, 3)), ratio=self.dropout_ratio)
        return transform_params
