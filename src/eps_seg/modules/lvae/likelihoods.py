import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from typing import Type, Union
# from e2cnn.nn import R2Conv, FieldType


class LikelihoodModule(nn.Module):

    def distr_params(self, x):
        pass

    @staticmethod
    def mean(params):
        pass

    @staticmethod
    def mode(params):
        pass

    @staticmethod
    def sample(params):
        pass

    def log_likelihood(self, x, params):
        pass

    def forward(self, input_, x):
        distr_params = self.distr_params(input_)
        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        if x is None:
            ll = None
        else:
            ll = self.log_likelihood(x, distr_params)
        dct = {
            "mean": mean,
            "mode": mode,
            "sample": sample,
            "params": distr_params,
        }
        return ll, dct


class GaussianLikelihood(LikelihoodModule):

    def __init__(self, ch_in, color_channels, conv_mult=2, r2_act=None):
        super().__init__()

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f"Conv{conv_mult}d")
        self.parameter_net = conv_type(ch_in, color_channels, kernel_size=3, padding=1)

    def distr_params(self, x):
        x = self.parameter_net(x)
        # mean, lv = x.chunk(2, dim=1)
        mean = x
        lv = None
        params = {
            "mean": mean,
            "logvar": lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params["mean"]

    @staticmethod
    def mode(params):
        return params["mean"]

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params["mean"]

    def log_likelihood(self, x, params):
        logprob = -0.5 * (params["mean"] - x) ** 2
        #         logprob = log_normal(x, params['mean'], params['logvar'], reduce='none')
        return logprob


def log_normal(x, mean, logvar, reduce="mean"):
    """
    Log of the probability density of the values x untder the Normal
    distribution with parameters mean and logvar. The sum is taken over all
    dimensions except for the first one (assumed to be batch). Reduction
    is applied at the end.
    :param x: tensor of points, with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param logvar: tensor with log-variance of distribution, shape has to be
                   either scalar or broadcastable
    :param reduce: reduction over batch: 'mean' | 'sum' | 'none'
    :return:
    """

    logvar = _input_check(x, mean, logvar, reduce)
    var = torch.exp(logvar)
    log_prob = -0.5 * (
        ((x - mean) ** 2) / var + logvar + torch.tensor(2 * math.pi).log()
    )
    log_prob = log_prob.sum((1, 2, 3))
    return _reduce(log_prob, reduce)
