import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.utils.rnn_factory import rnn_factory

class RNNUncondDecoder(DecoderBase):
    """Base recurrent attention-based decoder class.

    Specifies the interface used by different decoder types
    and required by :class:`~onmt.models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[memory_bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :class:`~onmt.modules.GlobalAttention`
       attn_func (str) : see :class:`~onmt.modules.GlobalAttention`
       coverage_attn (str): see :class:`~onmt.modules.GlobalAttention`
       context_gate (str): see :class:`~onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
       reuse_copy_attn (bool): reuse the attention for copying
       copy_attn_type (str): The copy attention style. See
        :class:`~onmt.modules.GlobalAttention`.
    """

    def __init__(self, rnn_type, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNUncondDecoder, self).__init__(attentional=False)
        if rnn_type == 'GRU':
            raise NotImplementedError

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.dropout,
            embeddings)

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        batch_size = memory_bank.shape[1]
        weight = next(self.parameters())
        h = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        c = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        self.state["hidden"] = (h, c)

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None, **kwargs):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        emb = self.embeddings(tgt)
        dec_outs, dec_state = self.rnn(emb, self.state["hidden"])
        dec_outs = self.dropout(dec_outs)

        self.state["hidden"] = dec_state

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

        return dec_outs, None
    
    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        return self.embeddings.embedding_size

