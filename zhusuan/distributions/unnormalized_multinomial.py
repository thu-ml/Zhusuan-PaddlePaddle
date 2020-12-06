
import paddle
import paddle.fluid as fluid

import numpy as np
import scipy as sp
import math

from .base import Distribution
from .utils import log_combination

__all__ = [
    'UnnormalizedMultinomial',
    'BagofCategoricals'
]


class UnnormalizedMultinomial(Distribution):
    def __init__(self,
                 dtype='int32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(UnnormalizedMultinomial, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._probs = kwargs['probs']
        self._n_categories = self._probs.shape[-1]
        self._normalize_logits = True
        if (not kwargs['normalize_logits'] is None) and \
                type(kwargs['normalize_logits']) == type(True):
            self._normalize_logits = kwargs['normalize_logits'] # Bool Value

    @property
    def probs(self):
        """The un-normalized log probabilities."""
        return self._probs

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    @property
    def normalize_logits(self):
        """A bool indicating whether `logits` should be
        normalized when computing probability"""
        return self._normalize_logits


    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        raise self._probs.shape[:-1]

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return self._probs.shape[:-1]

    def _sample(self, n_samples=1, **kwargs):
        # TODO: Here the error settings is the same as the one in TF version.
        raise NotImplementedError("Unnormalized multinomial distribution"
                                  " does not support sampling because"
                                  " n_experiments is not given. Please use"
                                  " class Multinomial to sample")


    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache
        sample = paddle.cast(sample, self.param_dtype)
        _probs = self.probs

        ## Log Prob
        if self.normalize_logits:
            _probs = _probs - paddle.logsumexp(_probs, axis=-1, keepdim=True)

        log_prob = sample * _probs

        # TODO: Check if reduce_sum is required here.
        log_prob = fluid.layers.reduce_sum(log_prob, dim=-1)

        return log_prob


BagofCategoricals = UnnormalizedMultinomial


