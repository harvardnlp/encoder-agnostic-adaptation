""" Embeddings module """
import math
import warnings

import torch
import torch.nn as nn

from onmt.modules.util_class import Elementwise
#from onmt.encoders.transformer import TransformerEncoder
import onmt.encoders
#from onmt.decoders.transformer import TransformerDecoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1) # [max_len, 1, dim]
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None, offset=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        if offset is not None:
            raise AssertionError

        emb = emb * math.sqrt(self.dim)

        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, context_size, embedding_dim, dropout=0):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(context_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, step=None, offset=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        if step is None:
            position_ids = torch.arange(0, emb.shape[0], dtype=torch.long, device=emb.device)
        else:
            position_ids = torch.arange(step, step+1, dtype=torch.long, device=emb.device)
        position_ids = position_ids.unsqueeze(1).repeat(1, emb.shape[1]) # [seq_len, batch_size]

        if offset is not None:
            offset = offset.unsqueeze(0) # [1, batch_size]
            position_ids += offset

        pe_vals = self.pe(position_ids) # [seq_len, batch_size, self.dim]
        emb = emb + pe_vals
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 position_encoding_learned=False,
                 position_encoding_ctxsize=1024,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False,
                 fix_word_vecs=False,
                 GPT_representation_mode='none',
                 GPT_representation_tgt=False):
        self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
                            feat_vec_size, feat_padding_idx)

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                      for vocab, dim, pad in emb_params]

        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        self.position_encoding = position_encoding

        if self.position_encoding:
            if position_encoding_learned:
                pe = LearnedPositionalEncoding(position_encoding_ctxsize, self.embedding_size, dropout=dropout)
                if fix_word_vecs:
                    pe.pe.weight.requires_grad = False
            else:
                pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)
            
        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False

        self.GPT_representation_mode = GPT_representation_mode
        self.GPT_representation_tgt = GPT_representation_tgt
        if self.GPT_representation_mode != 'none':
            gpt_dropout = 0 if self.GPT_representation_mode == 'elmo' else dropout
            if self.GPT_representation_tgt:
                self.gpt_model = onmt.decoders.TransformerDecoder(12, 768, 12, 3072, False, 'scaled-dot', gpt_dropout, gpt_dropout, None, 0, False, True, False, False)
            else:
                self.gpt_model = onmt.encoders.TransformerEncoder(12, 768, 12, 3072, gpt_dropout, gpt_dropout, None, 0, True)
            if self.GPT_representation_mode == 'elmo':
                for p in self.gpt_model.parameters():
                    p.requires_grad = False
                self.elmo_scale_params = nn.Parameter(torch.ones(13))
                self.elmo_gamma_param = nn.Parameter(torch.full((1,), 1.0))


    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                raise ValueError("Using feat_vec_exponent to determine "
                                 "feature vec size, but got feat_vec_exponent "
                                 "less than or equal to 0.")
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError("Got unequal number of feat_vocab_sizes and "
                             "feat_padding_idx ({:d} != {:d})".format(
                                n_feats, len(feat_padding_idx)))

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                    .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def forward(self, source, step=None, offset=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """

        emb = source
        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    emb = module(emb, step=step, offset=offset)
                else:
                    emb = module(emb)
        else:
            emb = self.make_embedding(emb)

        if self.GPT_representation_mode != 'none':
            if self.GPT_representation_tgt and step == 0:
                # Need to initialize cache for self attn layers
                self.gpt_model._init_cache(torch.zeros((source.shape[0], source.shape[1], 1), dtype=emb.dtype, device=emb.device))
                self.gpt_model.state['src'] = None

            words = source[:, :, 0].transpose(0, 1)
            w_batch, w_len = words.size()
            mask = words.data.eq(self.word_padding_idx).unsqueeze(1)  # [B, 1, T]

            if self.GPT_representation_mode == 'elmo':
                layer_weights = nn.functional.softmax(self.elmo_scale_params, dim=0)
                elmo_representation = layer_weights[0]*emb.transpose(0, 1).contiguous()

            # Run the forward pass of every layer of the tranformer.
            out = emb.transpose(0, 1).contiguous()
            for layer_num, layer in enumerate(self.gpt_model.transformer_layers):
                if self.GPT_representation_tgt:
                    layer_cache = self.gpt_model.state["cache"]["layer_{}".format(layer_num)] \
                        if step is not None else None
                    out, _ = layer(out, None, None, mask, layer_cache=layer_cache, step=step)
                else:
                    out = layer(out, mask)

                if self.GPT_representation_mode == 'elmo':
                    elmo_representation += layer_weights[layer_num+1]*out

            if self.GPT_representation_mode == 'elmo':
                emb = self.elmo_gamma_param*elmo_representation.transpose(0, 1).contiguous()
            else:
                emb = out.transpose(0, 1).contiguous()

        return emb
