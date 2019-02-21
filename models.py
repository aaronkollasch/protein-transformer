from typing import Union, Dict, Sequence
import warnings
from copy import deepcopy
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import recursive_update
import layers


class BaseModel(nn.Module):
    """Abstract RNN class."""
    MODEL_TYPE = 'abstract_model'
    DEFAULT_DIMS = {
            "batch": 10,
            "alphabet": 21,
            "length": 256
        }
    DEFAULT_PARAMS = {}

    def __init__(self, dims=None, hyperparams=None):
        nn.Module.__init__(self)

        self.dims = self.DEFAULT_DIMS.copy()
        if dims is not None:
            self.dims.update(dims)
        self.dims.setdefault('input', self.dims['alphabet'])

        self.hyperparams: Dict[str, Dict[str, Union[int, bool, float, str, Sequence]]] = deepcopy(self.DEFAULT_PARAMS)
        if hyperparams is not None:
            recursive_update(self.hyperparams, hyperparams)

        self.step = 0
        self.image_summaries = {}

    def forward(self, *args):
        raise NotImplementedError

    def weight_costs(self):
        raise NotImplementedError

    def weight_cost(self):
        return torch.stack(self.weight_costs()).sum()

    def parameter_count(self):
        return sum(param.numel() for param in self.parameters())


class Transformer(BaseModel):
    MODEL_TYPE = 'transformer'
    DEFAULT_PARAMS = {
            'transformer': {
                'd_model': 512,
                'd_ff': 2048,
                'num_heads': 8,
                'num_layers': 6,
                'dropout_p': 0.1,
            },
            'sampler_hyperparams': {
                'warm_up': 10000,
                'annealing_type': 'linear',
                'anneal_kl': True,
                'anneal_noise': True
            },
            'regularization': {
                'l2': True,
                'l2_lambda': 1.,
                'bayesian': False,  # if True, disables l2 regularization
                'bayesian_lambda': 0.1,
                'rho_type': 'log_sigma',  # lin_sigma, log_sigma
                'prior_type': 'gaussian',  # gaussian, scale_mix_gaussian
                'prior_params': None,
                'label_smoothing': True,
                'label_smoothing_eps': 0.1
            },
            'optimization': {
                'optimizer': 'Adam',
                'lr': 0,
                'lr_factor': 2,
                'lr_warmup': 4000,
                'weight_decay': 0,  # trainer will divide by n_eff before using
                'clip': 10.0,
                'opt_params': {
                    'betas': (0.9, 0.98),
                    'eps': 1e-9,
                },
                'lr_schedule': True,
            }
        }

    def __init__(self, dims=None, hyperparams=None):
        BaseModel.__init__(self, dims=dims, hyperparams=hyperparams)

        if self.hyperparams['regularization']['bayesian']:
            self.hyperparams['regularization']['l2'] = False
        elif self.hyperparams['regularization']['l2']:
            # torch built-in weight decay is more efficient than manual calculation
            self.hyperparams['optimization']['weight_decay'] = self.hyperparams['regularization']['l2_lambda']

        if self.hyperparams['regularization']['prior_params'] is None:
            if self.hyperparams['regularization']['prior_type'] == 'gaussian':
                self.hyperparams['regularization']['prior_params'] = (0., 1.)
            elif self.hyperparams['regularization']['prior_type'] == 'scale_mix_gaussian':
                self.hyperparams['regularization']['prior_params'] = (0.1, 0., 1., 0., 0.001)

        if self.hyperparams['regularization']['bayesian'] and self.hyperparams['transformer']['dropout_p'] > 0:
            warnings.warn("Using both weight uncertainty and dropout")

        params = self.hyperparams['transformer']
        n = params['num_layers']
        d_model = params['d_model']
        d_ff = params['d_ff']
        h = params['num_heads']
        dropout = params['dropout_p']
        c = deepcopy
        attn = layers.MultiHeadedAttention(h, d_model, dropout)
        ff = layers.PositionwiseFeedForward(d_model, d_ff, dropout)
        position = layers.PositionalEncoding(d_model, dropout)

        self.encoder = layers.Encoder(layers.EncoderLayer(d_model, c(attn), c(ff), dropout), n)
        self.decoder = layers.Decoder(layers.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n)
        self.src_embed = nn.Sequential(layers.Embeddings(d_model, self.dims['input']), c(position))
        self.tgt_embed = nn.Sequential(layers.Embeddings(d_model, self.dims['input']), c(position))
        self.h_to_out = nn.Linear(d_model, self.dims['alphabet'])
        self.reset_parameters()

        if self.hyperparams['regularization']['label_smoothing']:
            self.criterion = layers.LabelSmoothing(
                self.dims['alphabet'],
                smoothing=self.hyperparams['regularization']['label_smoothing_eps'],
                reduction='none')
        else:
            self.criterion = None

    def reset_parameters(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters') and m is not self:
                m.reset_parameters()
        # This was important from their code. Initialize parameters with Glorot or fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def weight_costs(self):
        return []
        # return (
        #     self.rnn.weight_costs() +
        #     [cost
        #      for name, module in self.dense_net_modules.items()
        #      if name.startswith('linear') or name.startswith('norm')
        #      for cost in module.weight_costs()]
        # )

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src: tensor(batch, src_length, in_channels)
        :param tgt: tensor(batch, tgt_length, out_channels)
        :param src_mask: tensor(batch, src_length, 1)
        :param tgt_mask: tensor(batch, tgt_length, 1)
        :return:
        """
        return self.h_to_out(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @staticmethod
    def reconstruction_loss(seq_logits, target_seqs, mask):
        seq_reconstruct = F.log_softmax(seq_logits, 2)
        cross_entropy = F.nll_loss(seq_reconstruct.transpose(1, 2), target_seqs.argmax(2), reduction='none')
        # cross_entropy = F.cross_entropy(seq_logits.transpose(1, 2), target_seqs.argmax(2), reduction='none')
        cross_entropy = cross_entropy * mask.squeeze(2)
        ce_loss_per_seq = cross_entropy.sum(1)
        bitperchar_per_seq = ce_loss_per_seq / mask.sum([1, 2])
        ce_loss = ce_loss_per_seq.mean()
        bitperchar = bitperchar_per_seq.mean()
        return {
            'seq_reconstruct': seq_reconstruct,
            'ce_loss': ce_loss,
            'ce_loss_per_seq': ce_loss_per_seq,
            'bitperchar': bitperchar,
            'bitperchar_per_seq': bitperchar_per_seq
        }

    def calculate_loss(
            self, seq_logits, target_seqs, mask, n_eff
    ):
        """

        :param seq_logits: (N, L, C)
        :param target_seqs: (N, L, C) as one-hot
        :param mask: (N, L, 1)
        :param n_eff:
        :param labels: (N, n_labels)
        :param pos_weight: (n_labels)
        :return:
        """
        reg_params = self.hyperparams['regularization']

        # cross-entropy
        reconstruction_loss = self.reconstruction_loss(
            seq_logits, target_seqs, mask
        )
        if reg_params['label_smoothing']:
            loss_per_seq = self.criterion(reconstruction_loss['seq_reconstruct'], target_seqs).sum([1, 2])
            loss = loss_per_seq.mean()
        else:
            loss_per_seq = reconstruction_loss['ce_loss_per_seq']
            loss = reconstruction_loss['ce_loss']

        # regularization
        if reg_params['bayesian']:
            loss += self.weight_cost() * reg_params['bayesian_lambda'] / n_eff
        elif reg_params['l2']:
            # # Skip; use built-in optimizer weight_decay instead
            # loss += self.weight_cost() * reg_params['l2_lambda'] / n_eff
            pass

        seq_reconstruct = reconstruction_loss.pop('seq_reconstruct')
        self.image_summaries['SeqReconstruct'] = dict(
            img=seq_reconstruct.transpose(1, 2).unsqueeze(3).detach(), max_outputs=3)
        self.image_summaries['SeqTarget'] = dict(
            img=target_seqs.transpose(1, 2).unsqueeze(3).detach(), max_outputs=3)
        self.image_summaries['SeqDelta'] = dict(
            img=(seq_reconstruct - target_seqs.transpose(1, 2)).transpose(1, 2).unsqueeze(3).detach(), max_outputs=3)

        output = {
            'loss': loss,
            'ce_loss': None,
            'bitperchar': None,
            'loss_per_seq': loss_per_seq,
            'bitperchar_per_seq': None,
            'ce_loss_per_seq': None
        }
        output.update(reconstruction_loss)
        return output


class TransformerDecoder(BaseModel):
    MODEL_TYPE = 'transformer_decoder'
    DEFAULT_PARAMS = {
            'transformer': {
                'd_model': 512,
                'd_ff': 2048,
                'num_heads': 8,
                'num_layers': 6,
                'dropout_p': 0.1,
            },
            'sampler_hyperparams': {
                'warm_up': 10000,
                'annealing_type': 'linear',
                'anneal_kl': True,
                'anneal_noise': True
            },
            'regularization': {
                'l2': True,
                'l2_lambda': 1.,
                'bayesian': False,  # if True, disables l2 regularization
                'bayesian_lambda': 0.1,
                'rho_type': 'log_sigma',  # lin_sigma, log_sigma
                'prior_type': 'gaussian',  # gaussian, scale_mix_gaussian
                'prior_params': None,
                'label_smoothing': True,
                'label_smoothing_eps': 0.1
            },
            'optimization': {
                'optimizer': 'Adam',
                'lr': 0,
                'lr_factor': 2,
                'lr_warmup': 4000,
                'weight_decay': 0,  # trainer will divide by n_eff before using
                'clip': 10.0,
                'opt_params': {
                    'betas': (0.9, 0.98),
                    'eps': 1e-9,
                },
                'lr_schedule': True,
            }
        }

    def __init__(self, dims=None, hyperparams=None):
        BaseModel.__init__(self, dims=dims, hyperparams=hyperparams)

        if self.hyperparams['regularization']['bayesian']:
            self.hyperparams['regularization']['l2'] = False
        elif self.hyperparams['regularization']['l2']:
            # torch built-in weight decay is more efficient than manual calculation
            self.hyperparams['optimization']['weight_decay'] = self.hyperparams['regularization']['l2_lambda']

        if self.hyperparams['regularization']['prior_params'] is None:
            if self.hyperparams['regularization']['prior_type'] == 'gaussian':
                self.hyperparams['regularization']['prior_params'] = (0., 1.)
            elif self.hyperparams['regularization']['prior_type'] == 'scale_mix_gaussian':
                self.hyperparams['regularization']['prior_params'] = (0.1, 0., 1., 0., 0.001)

        if self.hyperparams['regularization']['bayesian'] and self.hyperparams['transformer']['dropout_p'] > 0:
            warnings.warn("Using both weight uncertainty and dropout")

        params = self.hyperparams['transformer']
        n = params['num_layers']
        d_model = params['d_model']
        d_ff = params['d_ff']
        h = params['num_heads']
        dropout = params['dropout_p']
        c = deepcopy
        attn = layers.MultiHeadedAttention(h, d_model, dropout)
        ff = layers.PositionwiseFeedForward(d_model, d_ff, dropout)
        position = layers.PositionalEncoding(d_model, dropout)

        self.decoder = layers.Decoder(layers.DecoderLayer(d_model, c(attn), None, c(ff), dropout), n)
        self.tgt_embed = nn.Sequential(layers.Embeddings(d_model, self.dims['input']), c(position))
        self.h_to_out = nn.Linear(d_model, self.dims['alphabet'])
        self.reset_parameters()

        if self.hyperparams['regularization']['label_smoothing']:
            self.criterion = layers.LabelSmoothing(
                self.dims['alphabet'],
                smoothing=self.hyperparams['regularization']['label_smoothing_eps'],
                reduction='none')
        else:
            self.criterion = None

    def reset_parameters(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters') and m is not self:
                m.reset_parameters()
        # This was important from their code. Initialize parameters with Glorot or fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def weight_costs(self):
        return []
        # return (
        #     self.rnn.weight_costs() +
        #     [cost
        #      for name, module in self.dense_net_modules.items()
        #      if name.startswith('linear') or name.startswith('norm')
        #      for cost in module.weight_costs()]
        # )

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src: tensor(batch, src_length, in_channels) or None
        :param tgt: tensor(batch, tgt_length, out_channels)
        :param src_mask: tensor(batch, src_length, 1) or None
        :param tgt_mask: tensor(batch, tgt_length, 1)
        :return:
        """
        return self.h_to_out(self.decode(None, None, tgt, tgt_mask))

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @staticmethod
    def reconstruction_loss(seq_logits, target_seqs, mask):
        seq_reconstruct = F.log_softmax(seq_logits, 2)
        cross_entropy = F.nll_loss(seq_reconstruct.transpose(1, 2), target_seqs.argmax(2), reduction='none')
        # cross_entropy = F.cross_entropy(seq_logits.transpose(1, 2), target_seqs.argmax(2), reduction='none')
        cross_entropy = cross_entropy * mask.squeeze(2)
        ce_loss_per_seq = cross_entropy.sum(1)
        bitperchar_per_seq = ce_loss_per_seq / mask.sum([1, 2])
        ce_loss = ce_loss_per_seq.mean()
        bitperchar = bitperchar_per_seq.mean()
        return {
            'seq_reconstruct': seq_reconstruct,
            'ce_loss': ce_loss,
            'ce_loss_per_seq': ce_loss_per_seq,
            'bitperchar': bitperchar,
            'bitperchar_per_seq': bitperchar_per_seq
        }

    def calculate_loss(
            self, seq_logits, target_seqs, mask, n_eff,
    ):
        """

        :param seq_logits: (N, L, C)
        :param target_seqs: (N, L, C) as one-hot
        :param mask: (N, L, 1)
        :param n_eff:
        :return:
        """
        reg_params = self.hyperparams['regularization']

        # cross-entropy
        reconstruction_loss = self.reconstruction_loss(
            seq_logits, target_seqs, mask
        )
        if reg_params['label_smoothing']:
            loss_per_seq = self.criterion(reconstruction_loss['seq_reconstruct'], target_seqs).sum([1, 2])
            loss = loss_per_seq.mean()
        else:
            loss_per_seq = reconstruction_loss['ce_loss_per_seq']
            loss = reconstruction_loss['ce_loss']

        # regularization
        if reg_params['bayesian']:
            loss += self.weight_cost() * reg_params['bayesian_lambda'] / n_eff
        elif reg_params['l2']:
            # # Skip; use built-in optimizer weight_decay instead
            # loss += self.weight_cost() * reg_params['l2_lambda'] / n_eff
            pass

        seq_reconstruct = reconstruction_loss.pop('seq_reconstruct')
        self.image_summaries['SeqReconstruct'] = dict(
            img=seq_reconstruct.unsqueeze(3).detach(), max_outputs=3)
        self.image_summaries['SeqTarget'] = dict(
            img=target_seqs.unsqueeze(3).detach(), max_outputs=3)
        self.image_summaries['SeqDelta'] = dict(
            img=(seq_reconstruct - target_seqs).unsqueeze(3).detach(), max_outputs=3)

        output = {
            'loss': loss,
            'ce_loss': None,
            'bitperchar': None,
            'loss_per_seq': loss_per_seq,
            'bitperchar_per_seq': None,
            'ce_loss_per_seq': None
        }
        output.update(reconstruction_loss)
        return output
