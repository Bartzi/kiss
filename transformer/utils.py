import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer import Chain


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype(np.uint8)
    return mask == 0


class SublayerConnection(Chain):

    def __init__(self, layer, size, dropout_ratio=0.1):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.layer = layer
            self.norm = L.LayerNormalization(size)

    def __call__(self, x, layer):
        batch_size, M, size = x.shape
        normed_x = self.norm(F.reshape(x, (-1, size)))
        normed_x = F.reshape(normed_x, (batch_size, M, size))
        return x + F.dropout(layer(normed_x), ratio=self.dropout_ratio)
