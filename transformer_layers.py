# code adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions as dist

from utils import clones
import layers
from functions import Nonlinearity


class HyperparameterError(ValueError):
    pass


class EncoderDecoder(layers.Module):
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

    def weight_costs(self):
        return self.encoder.weight_costs() + self.decoder.weight_costs() + \
               self.src_embed.weight_costs() + self.tgt_embed.weight_costs + \
               self.generator.weight_costs()


class Generator(layers.Module):
    """Define standard linear + softmax generation step.
    """
    def __init__(self, d_model, vocab, dim=-1, hyperparams=None):
        super(Generator, self).__init__()
        if hyperparams and hyperparams['regularization']['bayesian']:
            linear = layers.BayesianLinear
        else:
            linear = layers.Linear

        self.proj = linear(d_model, vocab, hyperparams=hyperparams)
        self.dim = dim

    def forward(self, x):
        return F.log_softmax(self.proj(x, step=self.step), dim=self.dim)

    def weight_costs(self):
        return self.proj.weight_costs()


class Encoder(layers.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n, hyperparams=None):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)

        if hyperparams and hyperparams['regularization']['bayesian']:
            norm = layers.BayesianLayerNorm
        else:
            norm = layers.LayerNorm
        self.norm = norm(layer.size, hyperparams=hyperparams)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x, step=self.step)

    def weight_costs(self):
        return [p.pow(2).sum() for p in self.parameters()]


class SublayerConnection(layers.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout, hyperparams=None):
        super(SublayerConnection, self).__init__()
        if hyperparams and hyperparams['regularization']['bayesian']:
            norm = layers.BayesianLayerNorm
        else:
            norm = layers.LayerNorm
        self.norm = norm(size, g_init=1.0, bias_init=0.0, hyperparams=hyperparams)
        # TODO scale init by 1/sqrt(N) for N residual layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, residual=True):
        """Apply residual connection to any sublayer function that maintains the same size."""
        output = self.dropout(sublayer(self.norm(x, step=self.step)))
        if residual:
            output += x
        return output

    def weight_costs(self):
        return self.norm.weight_costs()


class EncoderLayer(layers.Module):
    """Encoder is made up of two sublayers, self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout, hyperparams=None):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, hyperparams=hyperparams), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, mask))
        return self.sublayer[1](x, self.feed_forward)

    def weight_costs(self):
        return self.self_attn.weight_costs() + self.feed_forward.weight_costs() + \
               [c for l in self.sublayer for c in l.weight_costs()]


class Decoder(layers.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, n, hyperparams=None):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)

        if hyperparams and hyperparams['regularization']['bayesian']:
            norm = layers.BayesianLayerNorm
        else:
            norm = layers.LayerNorm
        self.norm = norm(layer.size, hyperparams=hyperparams)

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
        return self.norm(x, self.step)

    def weight_costs(self):
        return [c for l in self.layers for c in l.weight_costs()] + self.norm.weight_costs()


class DecoderLayer(layers.Module):
    """Decoder is made up of three sublayers, self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, hyperparams=None):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, hyperparams=hyperparams), 2 + (src_attn is not None))

    def forward(self, x, memory, src_mask, tgt_mask, residual=True):
        m = memory
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, tgt_mask), residual=residual)
        if self.src_attn is not None:
            x = self.sublayer[1](x, lambda x_: self.src_attn(x_, m, m, src_mask))
        return self.sublayer[-1](x, self.feed_forward)

    def weight_costs(self):
        return (
            self.self_attn.weight_costs() +
            (self.src_attn.weight_costs() if self.src_attn is not None else []) +
            self.feed_forward.weight_costs() +
            [c for l in self.sublayer for c in l.weight_costs()]
        )


def attention(query, key, value, mask=None, dropout=0.0):
    """Compute 'Scaled Dot Product Attention'
    From http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)).div(math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(layers.Module):
    """Adapted from http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    def __init__(self, h, d_model, dropout=0.1, hyperparams=None):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        if d_model % h != 0:
            raise HyperparameterError(f"d_model {d_model} not divisible by num_heads {h}")
        # We assume d_v always equals d_k
        if hyperparams and hyperparams['regularization']['bayesian']:
            linear = layers.BayesianLinear
        else:
            linear = layers.Linear

        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(linear(d_model, d_model, hyperparams=hyperparams), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x, step=self.step).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = attention(query, key, value, mask=mask, dropout=self.p)
        self.attn = attn.detach()  # prevent from caching previous graph

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view((nbatches, -1, self.h * self.d_k))
        return self.linears[-1](x)

    def weight_costs(self):
        return [c for l in self.linears for c in l.weight_costs()]


def relative_attention(query, key, value, mask=None, dropout=0.0):
    """Compute 'Scaled Dot Product Attention'
    From http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)).div(math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedRelativeAttention(layers.Module):  # TODO implement relative attention
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
        x, attn = relative_attention(query, key, value, mask=mask, dropout=self.p)
        self.attn = attn.detach()  # prevent from caching previous graph

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view((nbatches, -1, self.h * self.d_k))
        return self.linears[-1](x)

    def weight_costs(self):
        return [c for l in self.linears for c in l.weight_costs()]


class PositionwiseFeedForward(layers.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1, nonlinearity='relu', hyperparams=None):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        if hyperparams and hyperparams['regularization']['bayesian']:
            linear = layers.BayesianLinear
        else:
            linear = layers.Linear
        self.w_1 = linear(d_model, d_ff, hyperparams=hyperparams)
        self.w_2 = linear(d_ff, d_model, hyperparams=hyperparams)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = Nonlinearity(nonlinearity)

    def forward(self, x):
        return self.w_2(self.dropout(self.nonlinearity(self.w_1(x, step=self.step))), step=self.step)

    def weight_costs(self):
        return self.w_1.weight_costs() + self.w_2.weight_costs()


class PositionalEncoding(layers.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000, random_start=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.random_start = random_start

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2).mul(-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.mul(div_term))
        pe[:, 1::2] = torch.cos(position.mul(div_term))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.random_start and self.training:
            start = np.random.randint(0, self.max_len-x.size(1))
            r = slice(start, start+x.size(1))
        else:
            r = slice(None, x.size(1))
        x = x + self.pe[:, r].requires_grad_(False)
        return self.dropout(x)

    def weight_costs(self):
        return []


class Embeddings(layers.Module):
    def __init__(self, d_model, vocab, hyperparams=None):
        super(Embeddings, self).__init__()
        if hyperparams and hyperparams['regularization']['bayesian']:
            linear = layers.BayesianLinear
        else:
            linear = layers.Linear

        self.lut = linear(vocab, d_model, hyperparams=hyperparams)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x, step=self.step) * math.sqrt(self.d_model)

    def weight_costs(self):
        return self.lut.weight_costs()


class LabelSmoothing(layers.Module):
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

    def weight_costs(self):
        return []
