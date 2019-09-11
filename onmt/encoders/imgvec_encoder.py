"""Define a minimal encoder."""
from onmt.encoders.encoder import EncoderBase
from torch import nn


class ImgVecEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, num_layers, emb_dim, outp_dim):
        super(ImgVecEncoder, self).__init__()
        self.num_layers = num_layers
        self.proj = nn.Linear(emb_dim, outp_dim)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
                opt.enc_layers,
                opt.image_channel_size,
                opt.word_vec_size)

    def forward(self, emb, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(emb, lengths)

        emb = self.proj(emb)
        _, batch, emb_dim = emb.size()

        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank, lengths
