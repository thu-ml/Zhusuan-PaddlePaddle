
import paddle
import paddle.fluid as fluid

import numpy as np
import scipy as sp
import math

from .base import Distribution
from .utils import open_interval_standard_uniform

__all__ = [
    'ExpConcrete',
    'ExpGumbelSoftmax'
]


class ExpConcrete(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(ExpConcrete, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._probs = kwargs['probs']
        self._n_categories = self._probs.shape[-1]
        self._temperature = kwargs['temperature']

    @property
    def probs(self):
        """The un-normalized log probabilities."""
        return self._probs

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    @property
    def temperature(self):
        """The temperature of ExpConcrete."""
        return self._temperature

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

        _probs, _temperature = self.probs, self.temperature
        if not self.is_reparameterized:
            _probs.stop_gradient = True
            _temperature.stop_gradient = True

        sample_shape_ = np.concatenate([[n_samples], self._probs.shape], axis=0).tolist()

        uniform = open_interval_standard_uniform(sample_shape_, self.dtype)
        gumbel = -paddle.log(-paddle.log(uniform))
        sample_ = paddle.nn.functional.log_softmax( (_probs + gumbel) / _temperature )

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache
        _probs, _temperature = self.probs, self.temperature

        ## Log Prob

        n = paddle.cast(self.n_categories, self.dtype)
        log_temperature = paddle.log(_temperature)

        temp = _probs - _temperature * sample
        # TODO: Paddle do not have lgamma module. Here we use Math package
        lgamma_n = paddle.to_tensor(list(map(lambda x: math.lgamma(x), n )))
        log_prob = lgamma_n + (n - 1) * log_temperature + \
                   fluid.layers.reduce_sum(temp, dim=-1) - n * paddle.logsumexp(temp, axis=-1)

        return log_prob

ExpGumbelSoftmax = ExpConcrete