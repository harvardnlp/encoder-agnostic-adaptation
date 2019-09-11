""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        device = key.device

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"].to(device), key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"].to(device), value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn

class JointMultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0, ctx_weight_param=False):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(JointMultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.ctx_linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.ctx_linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)

        if ctx_weight_param:
            self.ctx_bias = Parameter(torch.ones(1)*-10)
        self.ctx_weight_param = ctx_weight_param

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, self_kvq, ctx_kv, self_mask=None, ctx_mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           self_kvq (FloatTensor): set of `self_len`
               key vectors ``(batch, self_len, dim)``
           ctz_kv (FloatTensor): set of `ctx_len`
               value vectors ``(batch, ctx_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, self_len, self_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, self_len, dim)``
           * one of the attention vectors ``(batch, self_len, ctx_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = self_kvq.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        self_len = self_kvq.size(1)
        ctx_len = ctx_kv.size(1)
        device = self_kvq.device

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            query, self_key, self_value = self.linear_query(self_kvq),\
                                          self.linear_keys(self_kvq),\
                                          self.linear_values(self_kvq)
            #self_key = shape(self_key)
            #self_value = shape(self_value)
            if layer_cache["self_keys"] is not None:
                self_key = torch.cat(
                    (layer_cache["self_keys"].to(device), self_key),
                    dim=1)
            if layer_cache["self_values"] is not None:
                self_value = torch.cat(
                    (layer_cache["self_values"].to(device), self_value),
                    dim=1)
            layer_cache["self_keys"] = self_key
            layer_cache["self_values"] = self_value

            if layer_cache["memory_keys"] is None:
                ctx_key = self.ctx_linear_keys(ctx_kv) # [batch, ctx_len, dim]
                ctx_value = self.ctx_linear_values(ctx_kv)
                layer_cache["memory_keys"] = ctx_key
                layer_cache["memory_values"] = ctx_value
            else:
                ctx_key = layer_cache["memory_keys"]
                ctx_value = layer_cache["memory_values"]
        else:
            self_key = self.linear_keys(self_kvq) # [batch, self_len, dim]
            self_value = self.linear_values(self_kvq)
            query = self.linear_query(self_kvq)

            ctx_key = self.ctx_linear_keys(ctx_kv) # [batch, ctx_len, dim]
            ctx_value = self.ctx_linear_values(ctx_kv)

        self_len = self_key.size(1) # Need to do this again to include the layer_cache length 
        ctx_len = ctx_key.shape[1]

        key = torch.cat((self_key, ctx_key), dim=1)
        value = torch.cat((self_value, ctx_value), dim=1)

        key = shape(key)
        value = shape(value)

        if self.max_relative_positions > 0 and type == "self":
            raise NotImplementedError
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))

        query = shape(query)

        key_len = key.size(2) # self_len+ctx_len
        query_len = query.size(2) # self_len

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3)) # [batch, head, self_len, self_len+ctx_len]

        if self.ctx_weight_param:
            query_key[..., self_len:] += self.ctx_bias
        #print(query_key.mean(), query_key.std())

        if self.max_relative_positions > 0 and type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if self_mask is not None:
            self_mask = self_mask.unsqueeze(1)  # [B, 1, self_len, self_len]
            scores[:, :, :, :self_len] = scores[:, :, :, :self_len].masked_fill(self_mask, -1e18)
        if ctx_mask is not None:
            ctx_mask = ctx_mask.unsqueeze(1)  # [B, 1, 1, ctx_len]
            scores[:, :, :, self_len:] = scores[:, :, :, self_len:].masked_fill(ctx_mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value) # [batch, head, self_len, dim]

        if self.max_relative_positions > 0 and type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn (to context)
        ctx_attn_probs = attn[:, :, :, self_len:]
        ctx_attn_probs = ctx_attn_probs/ctx_attn_probs.sum(dim=-1, keepdim=True)

        top_attn = ctx_attn_probs \
            .view(batch_size, head_count,
                  query_len, ctx_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn, attn
