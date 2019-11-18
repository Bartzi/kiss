import chainer.functions as F
import chainer.links as L
import math

from chainer import Chain, initializers


class Embedding(Chain):

    def __init__(self, size, vocab_size):
        super().__init__()
        self.size = size

        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, size, initialW=initializers.GlorotUniform())

    def __call__(self, x):
        return self.embed(x) * math.sqrt(self.size)
