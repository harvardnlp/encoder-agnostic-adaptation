from onmt.encoders.encoder import EncoderBase

class EmbOnlyEncoder(EncoderBase):
    def __init__(self, embeddings):
        super(EmbOnlyEncoder, self).__init__()
        self.embeddings = embeddings

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(embeddings)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        emb = self.embeddings(src)
        return emb, emb, lengths
