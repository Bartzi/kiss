import chainer
import copy

from .encoder import Encoder, EncoderLayer
from .encoder_decoder import EncoderDecoder
from .decoder import DecoderLayer, Decoder
from .embedding import Embedding
from .positional_encoding import PositionalEncoding
from .position_wise_feed_forward import PositionwiseFeedForward
from .attention import MultiHeadedAttention


def build_decoder(vocab_size, N=6, model_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
    attention = MultiHeadedAttention(num_heads, model_size, dropout_ratio=dropout_ratio)
    feed_forward = PositionwiseFeedForward(model_size, ff_size, dropout_ratio=dropout_ratio)
    positional_encoding = PositionalEncoding(model_size, dropout_ratio=dropout_ratio)

    decoder_layer = DecoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(attention),
        feed_forward,
        dropout_ratio=dropout_ratio
    )

    decoder = Decoder(decoder_layer, N)

    embeddings = Embedding(model_size, vocab_size)

    return chainer.Sequential(embeddings, positional_encoding), decoder


def build_transform_param_decoder(N=1, model_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
    attention = MultiHeadedAttention(num_heads, model_size, dropout_ratio=dropout_ratio)
    feed_forward = PositionwiseFeedForward(model_size, ff_size, dropout_ratio=dropout_ratio)
    positional_encoding = PositionalEncoding(model_size, dropout_ratio=dropout_ratio)

    decoder_layer = DecoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(attention),
        feed_forward,
        dropout_ratio=dropout_ratio
    )

    decoder = Decoder(decoder_layer, N)

    return positional_encoding, decoder


def get_encoder_decoder(src_vocab_size, tgt_vocab_size, N=6, model_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
    attention = MultiHeadedAttention(num_heads, model_size, dropout_ratio=dropout_ratio)
    feed_forward = PositionwiseFeedForward(model_size, ff_size, dropout_ratio=dropout_ratio)
    positional_encoding = PositionalEncoding(model_size, dropout_ratio=dropout_ratio)

    encoder_layer = EncoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(feed_forward),
        dropout_ratio=dropout_ratio
    )
    encoder = Encoder(encoder_layer, N)

    decoder_layer = DecoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(attention),
        feed_forward,
        dropout_ratio=dropout_ratio
    )
    decoder = Decoder(decoder_layer, N)

    src_embeddings = Embedding(model_size, src_vocab_size)
    tgt_embeddings = Embedding(model_size, tgt_vocab_size)

    src_embeddings = chainer.Sequential(src_embeddings, positional_encoding)
    tgt_embeddings = chainer.Sequential(tgt_embeddings, positional_encoding)

    model = EncoderDecoder(
        encoder,
        decoder,
        src_embeddings,
        tgt_embeddings
    )

    return model


def get_conv_feature_encoder_decoder(vocab_size, N=6, model_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
    attention = MultiHeadedAttention(num_heads, model_size, dropout_ratio=dropout_ratio)
    feed_forward = PositionwiseFeedForward(model_size, ff_size, dropout_ratio=dropout_ratio)
    positional_encoding = PositionalEncoding(model_size, dropout_ratio=dropout_ratio)

    encoder_layer = EncoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(feed_forward),
        dropout_ratio=dropout_ratio
    )
    encoder = Encoder(encoder_layer, N)

    decoder_layer = DecoderLayer(
        model_size,
        copy.deepcopy(attention),
        copy.deepcopy(attention),
        feed_forward,
        dropout_ratio=dropout_ratio
    )
    decoder = Decoder(decoder_layer, N)

    embeddings = Embedding(model_size, vocab_size)

    tgt_embeddings = chainer.Sequential(embeddings, positional_encoding)
    src_embeddings = positional_encoding

    model = EncoderDecoder(
        encoder,
        decoder,
        src_embeddings,
        tgt_embeddings
    )

    return model
