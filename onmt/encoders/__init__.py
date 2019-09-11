"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders.imgvec_encoder import ImgVecEncoder
from onmt.encoders.embonly import EmbOnlyEncoder

class NoneEncoder:
    @classmethod
    def from_opt(cls, opt, embeddings):
        return None

str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder,
           "embonly": EmbOnlyEncoder, 'imgvec': ImgVecEncoder,
           'none': NoneEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc", "EmbOnlyEncoder"]
