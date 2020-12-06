import paddle
import paddle.fluid as fluid

import numpy as np
import math

from .base import Distribution
from .utils import open_interval_standard_uniform

__all__ = [
    'BinConcrete',
    'BinGumbelSoftmax'
]


class BinConcrete(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=False,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(BinConcrete, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)


        self._probs = kwargs['probs']
        self._temperature = kwargs['temperature']

    @property
    def probs(self):
        """The log-odds of probabilities."""
        return self._probs

    @property
    def temperature(self):
        """The temperature of BinConcrete."""
        return self._temperature

    def _batch_shape(self):
        raise self.probs.shape

    def _get_batch_shape(self):
        return self.probs.shape

    def _sample(self, n_samples=1, **kwargs):

        _probs, _temperature = self.probs, self.temperature

        if not self.is_reparameterized:
            _probs.stop_gradient = True
            _temperature.stop_gradient = True

        sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        uniform = open_interval_standard_uniform(sample_shape_, self.dtype)
        # TODO: add Logistic distribution
        logistic = paddle.log(uniform) - paddle.log(1 - uniform)
        sample_ = paddle.nn.functional.sigmoid((_probs + logistic) / _temperature)

        self.sample_cache = sample_
        assert (sample_.shape[0] == n_samples)
        return sample_


    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        _probs, _temperature = self.probs, self.temperature

        if len(sample.shape) > len(self._probs.shape):
            _probs = paddle.tile(_probs, repeat_times=\
                        [sample.shape[0], *len(_probs.shape)*[1]])
            _temperature = paddle.tile(_temperature, repeat_times=\
                        [sample.shape[0], *len(_temperature.shape)*[1]])
        else:
            _probs = self._probs
            _temperature = self._temperature

        ## Log Prob
        log_given = paddle.log(sample)
        log_1_minus_given = paddle.log(1 - sample)
        log_temperature = paddle.log(_temperature)

        logistic_given = log_given - log_1_minus_given
        temp = _temperature * logistic_given - _probs
        log_prob = log_temperature - log_given - log_1_minus_given + \
                   temp - 2 * paddle.nn.functional.softplus(temp)

        return log_prob


BinGumbelSoftmax = BinConcrete