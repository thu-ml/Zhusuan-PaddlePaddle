
import paddle
import paddle.fluid as fluid

import numpy as np
import scipy as sp
import math

from .base import Distribution
from .utils import log_combination

__all__ = [
    'Multinomial',
]


class Multinomial(Distribution):
    def __init__(self,
                 dtype='int32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(Multinomial, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._probs = kwargs['probs']
        self._n_categories = self._probs.shape[-1]
        self._n_experiments = kwargs['n_experiments']
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
    def n_experiments(self):
        """The number of experiments for each sample."""
        return self._n_experiments

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

        if self.n_experiments is None:
            raise ValueError('Cannot sample when `n_experiments` is None')

        if len(self._probs.shape) == 2:
            probs_flat = self._probs
        else:
            probs_flat = paddle.reshape(self._probs, [-1, self.n_categories])

        cate_ = paddle.distribution.Categorical(probs_flat)
        sample_flat_ = paddle.cast(cate_.sample([n_samples * self.n_experiments]), self.dtype)
        sample_shape_ = np.concatenate([[n_samples, self.n_experiments], self.batch_shape], axis=0).tolist()
        sample_ = paddle.reshape(sample_flat_, sample_shape_)
        sample_ = paddle.cast(paddle.nn.functional.one_hot(
            sample_, num_classes=self.n_categories), dtype=self.dtype)

        # TODO: Check if reduce_sum is required here.
        sample_ = fluid.layers.reduce_sum(sample_, dim=1)

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_


    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache
        sample = paddle.cast(sample, self.param_dtype)
        _probs = self.probs

        ## Log Prob
        if self.normalize_logits:
            _probs = _probs - paddle.logsumexp(_probs, axis=-1, keepdim=True)
        if self.n_experiments is None:
            n =fluid.layers.reduce_sum(sample, -1)
        else:
            n = paddle.cast(self.n_experiments, self.param_dtype)

        # TODO: Check if reduce_sum is required here.
        log_prob = log_combination(n, sample) + fluid.layers.reduce_sum(sample * _probs, dim=-1)

        return log_prob

