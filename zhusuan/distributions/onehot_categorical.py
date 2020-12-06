
import paddle
import paddle.fluid as fluid

import numpy as np
import scipy as sp
import math

from .base import Distribution
from .utils import log_combination

__all__ = [
    'OnehotCategorical',
    'OnehotDiscrete'
]


class OnehotCategorical(Distribution):
    def __init__(self,
                 dtype='int32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(OnehotCategorical, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._probs = kwargs['probs']
        self._n_categories = self._probs.shape[-1]

    @property
    def probs(self):
        """The un-normalized log probabilities."""
        return self._probs

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

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

        if len(self._probs.shape) == 2:
            probs_flat = self._probs
        else:
            probs_flat = paddle.reshape(self._probs, [-1, self.n_categories])

        cate_ = paddle.distribution.Categorical(probs_flat)
        sample_flat_ = paddle.cast(cate_.sample([n_samples]), self.dtype)

        if len(self._probs.shape) == 2:
            sample_ = sample_flat_
        else:
            sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
            sample_ = paddle.reshape(sample_flat_, sample_shape_)

        sample_ = paddle.cast(paddle.nn.functional.one_hot(sample_, self.n_categories),
                              dtype=self.dtype)

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache
        sample = paddle.cast(sample, self.param_dtype)
        _probs = self.probs

        ## Log Prob
        if (len(sample.shape) == 2) or (len(_probs.shape) == 2):
            sample_flat = sample
            logits_flat = _probs
        else:
            sample_flat = paddle.reshape(sample, [-1, self.n_categories])
            logits_flat = paddle.reshape(_probs, [-1, self.n_categories])

        # `label` type to calculate `softmax_with_cross_entropy` in PaddlePaddle
        # must be double
        sample_flat = paddle.cast(sample_flat, 'double')

        log_p_flat = -fluid.layers.reduce_sum(paddle.nn.functional.softmax_with_cross_entropy(
            label=sample_flat, logits=logits_flat, soft_label=True), dim=-1)

        if (len(sample.shape) == 2) or (len(_probs.shape) == 2):
            log_prob = log_p_flat
        else:
            log_prob = paddle.reshape(log_p_flat, _probs.shape[:-1])

        return log_prob



OnehotDiscrete = OnehotCategorical

