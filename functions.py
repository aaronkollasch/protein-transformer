import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import torch.autograd as autograd


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


def gelu_bert(x):
    """BERT's implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x.mul(0.5) * torch.erf(x / math.sqrt(2.0)).add(1.0)


def gelu_gpt(x):
    return x.mul(0.5) * torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x.pow(3))).add(1.0)


gelu = gelu_gpt


def swish(x):
    return x * torch.sigmoid(x)


def log_one_plus_exp(x):
    """Numerically stable log(1+exp(x)).
    Equivalent to -F.logsigmoid(-x).
    """
    a = x.clamp_min(0.)
    return a + torch.log((-a).exp() + (x - a).exp())


ACT_TO_FUN = {
    'elu': F.elu,
    'relu': F.relu,
    'lrelu': F.leaky_relu,
    'gelu': gelu,
    'gelu_bert': gelu_bert,
    'gelu_gpt': gelu_gpt,
    'swish': swish,
    'log_one_plus_exp': log_one_plus_exp,
    'none': lambda x: x,
}


def nonlinearity(nonlin_type):
    return ACT_TO_FUN[nonlin_type]


class Nonlinearity(nn.Module):
    def __init__(self, nonlin_type):
        nn.Module.__init__(self)
        self.nonlin_type = nonlin_type
        self.nonlinearity = nonlinearity(nonlin_type)

    def forward(self, x):
        return self.nonlinearity(x)

    def extra_repr(self):
        return f'"{self.nonlin_type}"'


def comb_losses(losses_f, losses_r):
    losses_comb = {}
    for key in losses_f.keys():
        if 'per_seq' in key:
            losses_comb[key] = torch.stack((losses_f[key], losses_r[key]))
        else:
            losses_comb[key] = losses_f[key] + losses_r[key]
            losses_comb[key + '_f'] = losses_f[key]
            losses_comb[key + '_r'] = losses_r[key]
    return losses_comb


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.tril(np.ones(attn_shape), k=0).astype('uint8')
    return torch.as_tensor(mask)  # TODO handle more shapes than NLC


def random_subsequent_mask(size):
    """Mask out randomly ordered subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.tril(np.ones(attn_shape), k=0).astype('uint8')
    order = np.arange(size)
    np.random.shuffle(order)
    mask = mask[:, :, order][:, order]
    mask = np.ascontiguousarray(mask)  # mask should already be contiguous, but statement copies only if necessary.
    return torch.as_tensor(mask)


def diagonal_mask(size):
    """Mask out diagonal positions."""
    return torch.diag(torch.ones(size)).unsqueeze(0) == 0  # TODO handle more shapes than NLC


def make_std_mask(src, tgt, l_dim=1, c_dim=2):
    src_mask = tgt_mask = None
    if src is not None:
        src_mask = (src.sum(c_dim) != 0).unsqueeze(l_dim)
    if tgt is not None:
        tgt_mask = (tgt.sum(c_dim) != 0).unsqueeze(l_dim)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(l_dim)).type_as(tgt_mask.data)
    return src_mask, tgt_mask


def make_1d_mask(src, l_dim=1, c_dim=2):
    if src is not None:
        return (src.sum(c_dim) != 0).unsqueeze(l_dim)


def make_2d_mask(tgt, random_order=False, l_dim=1, c_dim=2):
    if tgt is not None:
        tgt_mask = (tgt.sum(c_dim) != 0).unsqueeze(l_dim)
        if random_order:
            tgt_mask = tgt_mask & random_subsequent_mask(tgt.size(l_dim)).type_as(tgt_mask.data)
        else:
            tgt_mask = tgt_mask & subsequent_mask(tgt.size(l_dim)).type_as(tgt_mask.data)
        return tgt_mask


def make_1d_to_2d_mask(mask, mask_diag=True, l_dim=1, c_dim=2):
    if mask_diag:
        return mask & mask.transpose(l_dim, c_dim) & diagonal_mask(mask.size(l_dim)).type_as(mask.data)
    else:
        return mask & mask.transpose(l_dim, c_dim)


def clamp(x, min_val=0., max_val=1.):
    return max(min_val, min(x, max_val))


def l2_normalize(w, dim, eps=1e-12):
    """PyTorch implementation of tf.nn.l2_normalize
    """
    return w / w.pow(2).sum(dim, keepdim=True).clamp(min=eps).sqrt()


def l2_norm_except_dim(w, dim, eps=1e-12):
    norm_dims = [i for i, _ in enumerate(w.shape)]
    del norm_dims[dim]
    return l2_normalize(w, norm_dims, eps)


def moments(x, dim, keepdim=False):
    """PyTorch implementation of tf.nn.moments over a single dimension
    """
    # n = x.numel() / torch.prod(torch.tensor(x.shape)[dim])  # useful for multiple dims
    mean = x.mean(dim=dim, keepdim=True)
    variance = (x - mean.detach()).pow(2).mean(dim=dim, keepdim=keepdim)
    if not keepdim:
        mean = mean.squeeze(dim)
    return mean, variance


class Normalize(autograd.Function):
    """Normalize x across dim
    """
    @staticmethod
    def forward(ctx, x, dim, eps=1e-5):
        x_mu = x - x.mean(dim=dim, keepdim=True)
        inv_std = 1 / (x_mu.pow(2).mean(dim=dim, keepdim=True) + eps).sqrt()
        x_norm = x_mu * inv_std

        if ctx is not None:
            ctx.save_for_backward(x_mu, inv_std)
            ctx.dim = dim
        return x_norm

    @staticmethod
    def backward(ctx, grad_out):
        x_mu, inv_std = ctx.saved_tensors
        dim = ctx.dim
        n = x_mu.size(dim)

        # adapted from: https://cthorey.github.io/backpropagation/
        #               https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        dx = inv_std / n * (
                 grad_out * n -
                 grad_out.sum(dim, keepdim=True) -
                 (grad_out * x_mu).sum(dim, keepdim=True) * x_mu * inv_std ** 2
             )
        return dx, None, None

    @staticmethod
    def test():
        x = torch.DoubleTensor(2, 3, 1, 4).normal_(0, 1).requires_grad_()
        inputs = (x, 1)
        return autograd.gradcheck(Normalize.apply, inputs)


normalize = Normalize.apply


class NoamOpt:
    """Optimizer wrapper that implements rate.
    From http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer: optim.Optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, closure=None):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step(closure=closure)

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['step'] = self._step
        return state_dict

    def load_state_dict(self, state_dict):
        self._step = state_dict.get('step', self._step)
        return self.optimizer.load_state_dict(state_dict)


def anneal(step, hyperparams):
    warm_up = hyperparams["warm_up"]
    annealing_type = hyperparams["annealing_type"]
    if annealing_type == "linear":
        return min(step / warm_up, 1.)
    elif annealing_type == "piecewise_linear":
        return clamp(torch.tensor(step - warm_up).float().sigmoid().item() * ((step - warm_up) / warm_up))
    elif annealing_type == "sigmoid":
        slope = hyperparams["sigmoid_slope"]
        return torch.sigmoid(torch.tensor(slope * (step - warm_up))).item()


def rsample(mu, sigma, stddev=None):
    """Reparameterized sample from normal distribution"""
    eps = torch.randn_like(sigma)
    if stddev is not None:
        eps *= stddev
    return mu + sigma * eps


def kl_diag_gaussians(mu, logvar, prior_mu, prior_logvar):
    """ KL divergence between two Diagonal Gaussians """
    return 0.5 * (
        prior_logvar - logvar
        + (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp()
    ).sum() - 0.5


# def kl_standard_normal(mu, logvar):
#     """ KL divergence between Diagonal Gaussian and standard normal"""
#     return -0.5 * (logvar - mu.pow(2) - logvar.exp()).sum() - 0.5


def kl_standard_normal(mu, sigma):
    """ KL divergence between Diagonal Gaussian and standard normal"""
    return -0.5 * (sigma.log() * 2 - mu.pow(2) - sigma.pow(2)).sum() - 0.5


def kl_normal(mu, sigma, prior_mu=0., prior_sigma=1.):
    return dist.kl_divergence(
        dist.Normal(mu, sigma),
        dist.Normal(prior_mu, prior_sigma)
    )


def kl_mixture_gaussians(
        mu, sigma,
        p=0.1, mu_one=0., sigma_one=1., mu_two=0., sigma_two=0.001,
):
    prob_one = dist.Normal(mu_one, sigma_one).log_prob(mu) + math.log(clamp(p, 1e-10, 1))
    prob_two = dist.Normal(mu_two, sigma_two).log_prob(mu) + math.log(clamp(1 - p, 1e-10, 1))
    entropy = 0.5 * math.log(2.0 * math.pi * math.e) + sigma.log()
    return torch.logsumexp(torch.stack((prob_one, prob_two)), dim=0).add(entropy)


def mle_mixture_gaussians(w, p=0.1, mu_one=0., sigma_one=1., mu_two=0., sigma_two=0.01):
    """Log probability of w with a scale mixture of gaussians prior"""
    prob_one = dist.Normal(mu_one, sigma_one).log_prob(w) + math.log(clamp(p, 1e-10, 1))
    prob_two = dist.Normal(mu_two, sigma_two).log_prob(w) + math.log(clamp(1-p, 1e-10, 1))
    return torch.logsumexp(torch.stack((prob_one, prob_two)), dim=0)
