# code adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions as dist

from functions import attention
from utils import clones
from layers import LayerNorm


class HyperparameterError(ValueError):
    pass


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    From http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step.
    """
    def __init__(self, d_model, vocab, dim=-1):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.dim = dim

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=self.dim)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def weight_costs(self):
        return [p.pow(2).sum() for p in self.parameters()]


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size, g_init=1.0, bias_init=0.0)  # TODO scale init by 1/sqrt(N) for N residual layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, residual=True):
        """Apply residual connection to any sublayer function that maintains the same size."""
        output = self.dropout(sublayer(self.norm(x)))
        if residual:
            output += x
        return output


class EncoderLayer(nn.Module):
    """Encoder is made up of two sublayers, self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, residual='all'):
        for i, layer in enumerate(self.layers):
            if residual == 'none' or (i == 0 and residual == 'not_first'):
                residual = False
            else:
                residual = True

            if isinstance(tgt_mask, list):
                tgt_mask_i = tgt_mask[i]
            else:
                tgt_mask_i = tgt_mask

            x = layer(x, memory, src_mask, tgt_mask_i, residual=residual)
        return self.norm(x)

    def weight_costs(self):
        return [p.pow(2).sum() for p in self.parameters()]


class DecoderLayer(nn.Module):
    """Decoder is made up of three sublayers, self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2 + (src_attn is not None))

    def forward(self, x, memory, src_mask, tgt_mask, residual=True):
        m = memory
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, tgt_mask), residual=residual)
        if self.src_attn is not None:
            x = self.sublayer[1](x, lambda x_: self.src_attn(x_, m, m, src_mask))
        return self.sublayer[-1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """From http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        if d_model % h != 0:
            raise HyperparameterError(f"d_model {d_model} not divisible by num_heads {h}")
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedRelativeAttention(nn.Module):  # TODO implement relative attention
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedRelativeAttention, self).__init__()
        if d_model % h != 0:
            raise HyperparameterError(f"d_model {d_model} not divisible by num_heads {h}")
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2).mul(-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.mul(div_term))
        pe[:, 1::2] = torch.cos(position.mul(div_term))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, smoothing=0.0, c_dim=-1, reduction='batchmean'):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.c_dim = c_dim
        self.size = size

    def forward(self, x, target):
        if x.size(self.c_dim) != self.size:
            raise HyperparameterError(f"Channel size mismatch: {x.size(self.c_dim)} != {self.size}")
        true_dist = target.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.masked_fill_(target == 1, self.confidence)
        true_dist.masked_fill_(target.data.sum(self.c_dim, keepdim=True) == 0, 0.0)
        return self.criterion(x, true_dist.requires_grad_(False))
