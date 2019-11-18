import chainer
import chainer.functions as F
import chainer.links as L

from functions.rotation_droput import rotation_dropout
from text.text_localizer import TextLocalizer
from transformer import build_transform_param_decoder
from transformer.utils import subsequent_mask


class TransformerTextLocalizer(TextLocalizer):

    def __init__(self, *args, **kwargs):
        self.transformer_size = kwargs.pop('transformer_size', 512)

        super().__init__(*args, **kwargs)

        with self.init_scope():
            positional_encoding, decoder = build_transform_param_decoder(N=1, model_size=self.transformer_size)
            self.positional_encoding = positional_encoding
            self.decoder = decoder
            self.param_predictor = L.Linear(self.transformer_size, 6)

            params = self.param_predictor.b.data
            # params[...] = 0
            # params[[0, 4]] = 0.8
            # self.param_predictor.W.data[...] = 0

            self.param_embedder = L.Linear(6, self.transformer_size)
            self.mask = subsequent_mask(self.transformer_size)

    def get_transform_params(self, features):
        batch_size, num_channels, feature_height, feature_weight = features.shape
        features = F.reshape(features, (batch_size, num_channels, -1))
        features = F.transpose(features, (0, 2, 1))

        target = chainer.Variable(self.xp.zeros((batch_size, 1, 6), dtype=chainer.get_dtype()))

        for _ in range(self.num_bboxes_to_localize):
            embedded_params = self.param_embedder(target.array, n_batch_axes=2)
            embedded_params = self.positional_encoding(embedded_params)
            decoded = self.decoder(embedded_params, features, None, self.mask)
            params = self.param_predictor(decoded, n_batch_axes=2)
            target = F.concat([target, params[:, -1:]])

        target = F.reshape(target[:, 1:], (-1,) + target.shape[2:])
        transform_params = rotation_dropout(F.reshape(target, (-1, 2, 3)), ratio=self.dropout_ratio)
        return transform_params
