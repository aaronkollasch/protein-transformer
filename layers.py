import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions as dist

from functions import anneal, rsample, kl_normal, mle_mixture_gaussians, normalize


class HyperparameterError(ValueError):
    pass


class Module(nn.Module):
    """Abstract Module."""
    def __init__(self):
        nn.Module.__init__(self)
        self._step = 0

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        for m in self.modules():
            if isinstance(m, Module):
                m._step = step

    def reset_all_parameters(self):
        for m in self.modules():
            try:
                m.reset_parameters()
            except AttributeError:
                pass

    def reset_parameters(self):
        return

    def forward(self, *inputs):
        raise NotImplementedError

    def weight_costs(self):
        raise NotImplementedError

    def weight_cost(self):
        return torch.stack(self.weight_costs()).sum()

    def parameter_count(self):
        return sum(param.numel() for param in self.parameters())


class GaussianWeightUncertainty(object):
    """Implementation of weight uncertainty
    Paper: https://arxiv.org/abs/1505.05424
    See also https://www.nitarshan.com/bayes-by-backprop/"""
    def __init__(self, name, hyperparams):
        self.name = name
        self.rho_type = hyperparams['rho_type']  # lin_sigma, log_sigma
        self.prior_type = hyperparams['prior_type']  # gaussian, scale_mix_gaussian, laplacian
        self.prior_params = hyperparams['prior_params']

    def rho_to_sigma(self, rho):
        if self.rho_type == 'lin_sigma':
            return F.softplus(rho)
        elif self.rho_type == 'log_sigma':
            return rho.exp()

    def sample_weight(self, module, stddev=1.):
        mu = getattr(module, self.name + '_mu')
        if not module.training or stddev == 0.:
            return mu
        sigma = self.rho_to_sigma(getattr(module, self.name + '_rho'))
        return rsample(mu, sigma, stddev=stddev)

    def weight_costs(self, module):
        mu = getattr(module, self.name + '_mu')
        sigma = self.rho_to_sigma(getattr(module, self.name + '_rho'))
        if self.prior_type == 'gaussian':
            return kl_normal(mu, sigma, *self.prior_params)
        elif self.prior_type in ['scale_mix_gaussian', 'laplacian']:
            # No analytical solution for KL divergence,
            # so use current sample from variational posterior instead.
            if module.training:
                weight = getattr(module, self.name)
            else:
                weight = mu

            log_prior = None
            if self.prior_type == 'scale_mix_gaussian':
                log_prior = mle_mixture_gaussians(weight, *self.prior_params)
            elif self.prior_type == 'laplacian':
                log_prior = dist.Laplace(*self.prior_params).log_prob(weight)

            log_variational_posterior = dist.Normal(mu, sigma).log_prob(weight)
            return log_variational_posterior - log_prior

    def weight_cost(self, module):
        return self.weight_costs(module).sum()

    def apply(self, module):
        weight = getattr(module, self.name)
        del module._parameters[self.name]  # remove w from parameter list
        module.register_parameter(self.name + '_mu', Parameter(weight.data))
        module.register_parameter(self.name + '_rho', Parameter(-7 * torch.ones_like(weight).data))

        # Save detached Tensor during initialization to allow deepcopy of Tensor
        # during initialization with modules that use deepcopy.
        # Detached Tensor is replaced with attached Tensor during forward pass.
        object.__setattr__(module, self.name, self.sample_weight(module).detach())

    def remove(self, module):
        weight = getattr(module, self.name + '_mu')
        delattr(module, self.name)
        del module._parameters[self.name + '_mu']
        del module._parameters[self.name + '_rho']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, stddev=1.):
        object.__setattr__(module, self.name, self.sample_weight(module, stddev=stddev))


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        kwargs.pop('hyperparams', None)
        nn.Linear.__init__(self, *args, **kwargs)

    def weight_costs(self):
        return [self.weight.pow(2).sum(), self.bias.pow(2).sum()]

    def forward(self, x, step=0):
        return nn.Linear.forward(self, x)


class BayesianLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        nn.Linear.__init__(self, *args, **kwargs)

        self.regularizers = {}
        names = [name for name, _ in self.named_parameters()]
        for name in names:
            regularizer = GaussianWeightUncertainty(name, self.hyperparams['regularization'])
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_cost(self) for regularizer in self.regularizers.values()]

    def forward(self, x, step=0):
        stddev = anneal(step, self.hyperparams['sampler_hyperparams'])
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)

        output = nn.Linear.forward(self, x)
        return output


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        kwargs.pop('hyperparams', None)
        nn.BatchNorm1d.__init__(self, *args, **kwargs)

    def weight_costs(self):
        if self.affine:
            return [self.weight.pow(2).sum(), self.bias.pow(2).sum()]
        else:
            return []

    def forward(self, x, step=0):
        return nn.BatchNorm1d.forward(self, x)


class BayesianBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        nn.BatchNorm1d.__init__(self, *args, **kwargs)

        self.regularizers = {}
        names = [name for name, _ in self.named_parameters()]
        for name in names:
            regularizer = GaussianWeightUncertainty(name, self.hyperparams['regularization'])
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_cost(self) for regularizer in self.regularizers.values()]

    def forward(self, x, step=0):
        stddev = anneal(step, self.hyperparams['sampler_hyperparams'])
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)

        output = nn.BatchNorm1d.forward(self, x)
        return output


class LayerNorm(nn.Module):
    """Normalizes across dimension `dim`.
    """
    __constants__ = ['num_channels', 'dim', 'eps', 'affine', 'weight', 'bias', 'g_init', 'bias_init']

    def __init__(self, num_channels, dim=2, eps=1e-6, affine=True, g_init=1.0, bias_init=0.0, **_):
        super(LayerNorm, self).__init__()
        self.num_channels = num_channels
        self.dim = dim
        self.eps = eps
        self.affine = affine
        self.g_init = g_init
        self.bias_init = bias_init
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.normal_(self.weight, mean=self.g_init, std=1e-6)
            nn.init.normal_(self.bias, mean=self.bias_init, std=1e-6)

    def weight_costs(self):
        if self.affine:
            return [self.weight.pow(2).sum(), self.bias.pow(2).sum()]
        else:
            return []

    def forward(self, x, step=0):
        h = normalize(x, self.dim)

        if self.affine:
            shape = [1 for _ in x.shape]
            shape[self.dim] = self.num_channels
            h = self.weight.view(shape) * h + self.bias.view(shape)
        return h


class BayesianLayerNorm(LayerNorm):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        LayerNorm.__init__(self, *args, **kwargs)

        self.regularizers = {}
        names = [name for name, _ in self.named_parameters()]
        for name in names:
            regularizer = GaussianWeightUncertainty(name, self.hyperparams['regularization'])
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_cost(self) for regularizer in self.regularizers.values()]

    def forward(self, x, step=0):
        stddev = anneal(step, self.hyperparams['sampler_hyperparams'])
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)

        output = LayerNorm.forward(self, x, step=step)
        return output


class GRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('hyperparams', None)
        nn.GRU.__init__(self, *args, **kwargs)

    def weight_costs(self):
        return [param.pow(2).sum() for param in self.parameters()]

    def forward(self, x, hx=None, step=0):
        return nn.GRU.forward(self, x, hx)


class BayesianGRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        self.regularizers = {}
        nn.GRU.__init__(self, *args, **kwargs)

        names = [name for name, _ in self.named_parameters()]
        for name in names:
            regularizer = GaussianWeightUncertainty(name, self.hyperparams['regularization'])
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_cost(self) for regularizer in self.regularizers.values()]

    def forward(self, x, hx=None, step=0):
        stddev = anneal(step, self.hyperparams['sampler_hyperparams'])
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)

        with warnings.catch_warnings():
            # We recalculate the weights at every call so there's no point flattening; ignore this warning.
            warnings.simplefilter("ignore")
            output = nn.GRU.forward(self, x, hx)
        return output

    def flatten_parameters(self):
        # Prevents RuntimeError: _cudnn_rnn_flatten_weight not supported on torch.FloatTensor
        return

    @property
    def _flat_weights(self):
        # redefine necessary to prevent nn.GRU from using the wrong weights
        # fix included in PyTorch 1.0.1
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):  # included with _flat_weights for interpretability, but may not be necessary
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
