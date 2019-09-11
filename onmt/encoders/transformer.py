"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.gpt_mlp import MLP

class TransformerGPTEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
                 max_relative_positions=0):
        super(TransformerGPTEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attn_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = MLP(d_model, d_model*4, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        dec_mask = None
        src_len = mask.size(-1)
        future_mask = torch.ones(
            [src_len, src_len],
            device=mask.device,
            dtype=torch.uint8)
        future_mask = future_mask.triu_(1).view(1, src_len, src_len)
        dec_mask = torch.gt(mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=dec_mask, type="self")

        context = self.dropout(context) + inputs
        context_norm = self.layer_norm_2(context)

        output = self.feed_forward(context_norm)
        output = output + context

        return output

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attn_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attn_dropout, embeddings,
                 max_relative_positions, use_GPT_version):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings

        if use_GPT_version:
            layer_cls = TransformerGPTEncoderLayer
        else:
            layer_cls = TransformerEncoderLayer

        self.transformer_layers = nn.ModuleList(
            [layer_cls(
                d_model, heads, d_ff, dropout, attn_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.enc_heads,
            opt.transformer_ff,
            opt.dropout,
            opt.attn_dropout if hasattr(opt, 'attn_dropout') else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.enc_use_GPT_version)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer_layers:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
