from chainer import Chain


class EncoderDecoder(Chain):

    def __init__(self, encoder, decoder, src_embeddings, tgt_embeddings):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.src_embeddings = src_embeddings
            self.tgt_embeddings = tgt_embeddings

    def __call__(self, src, tgt, src_mask, tgt_mask):
        return self.decode(
            self.encode(src, src_mask),
            src_mask,
            tgt,
            tgt_mask
        )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embeddings(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embeddings(tgt), memory, src_mask, tgt_mask)

    @property
    def attention_maps(self):
        return {
            "encoder": self.encoder.attention_maps,
            "decoder": self.decoder.attention_maps,
        }
