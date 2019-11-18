import chainer.functions as F
import chainer.links as L

from chainer import Chain

from transformer.utils import SublayerConnection


class Decoder(Chain):

    def __init__(self, sublayer, N):
        super().__init__()
        with self.init_scope():
            self.sub_layers = sublayer.repeat(N, mode='copy')
            self.norm = L.LayerNormalization(sublayer.size)

    def __call__(self, x, memory, src_mask, tgt_mask):
        for layer in self.sub_layers:
            x = layer(x, memory, src_mask, tgt_mask)

        batch_size, num_steps, size = x.shape
        normed_x = self.norm(F.reshape(x, (-1, size)))
        return F.reshape(normed_x, (batch_size, num_steps, size))

    @property
    def attention_maps(self):
        return [layer.attention_maps for layer in self.sub_layers]


class DecoderLayer(Chain):

    def __init__(self, size, self_attention, src_attention, feed_forward, dropout_ratio=0.1):
        super().__init__()
        self.size = size
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.self_attention = SublayerConnection(self_attention, self.size, self.dropout_ratio)
            self.src_attention = SublayerConnection(src_attention, self.size, self.dropout_ratio)
            self.feed_forward = SublayerConnection(feed_forward, self.size, self.dropout_ratio)

    def __call__(self, x, memory, src_mask, tgt_mask):
        x = self.self_attention(x, lambda x: self.self_attention.layer(x, x, x, tgt_mask))
        x = self.src_attention(x, lambda x: self.src_attention.layer(x, memory, memory, src_mask))
        return self.feed_forward(x, lambda x: self.feed_forward.layer(x))

    @property
    def attention_maps(self):
        return {
            "self_attention": self.self_attention.layer.attention,
            "src_attention": self.src_attention.layer.attention,
        }
