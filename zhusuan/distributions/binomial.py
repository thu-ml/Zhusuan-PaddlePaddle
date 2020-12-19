import paddle
import paddle.fluid as fluid

import numpy as np
import math
from scipy import stats

from .base import Distribution

__all__ = [
    'Binomial'
]


class Binomial(Distribution):
    def __init__(self,
                 dtype='int32',
                 param_dtype='float32',
                 is_continues=False,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(Binomial, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)


        self._probs = kwargs['probs']
        # n_experiments must be positive
        self._n_experiments = kwargs['n_experiments']

    @property
    def probs(self):
        """The log-odds of probabilities."""
        return self._probs

    @property
    def n_experiments(self):
        """The number of experiments."""
        return self._n_experiments

    def _batch_shape(self):
        return self.probs.shape

    def _get_batch_shape(self):
        return self.probs.shape

    def _sample(self, n_samples=1, **kwargs):

        n = self.n_experiments

        _probs = self._probs
        if not self.is_reparameterized:
            _probs = _probs * 1
            _probs.stop_gradient = True

        if n_samples > 1:
            sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        else:
            sample_shape_ = self.batch_shape

        _probs *= paddle.less_equal(_probs, paddle.ones_like(_probs))

        # TODO: Paddle do not have poisson distribution module. Here we use Numpy Random
        sample_ = paddle.to_tensor(
            np.random.binomial(p=_probs.numpy(), n=self.n_experiments, size=sample_shape_))

        # # A different way to get samples:
        # if len(self.probs.shape) == 1:
        #     probs_flat = self._probs
        # else:
        #     probs_flat = paddle.reshape(self._probs, [-1])
        #
        # log_1_minus_p = -paddle.nn.functional.softplus(probs_flat)
        # log_p = probs_flat + log_1_minus_p
        # stacked_logits_flat = paddle.stack([log_1_minus_p, log_p], axis=-1)
        #
        # cate_ = paddle.distribution.Categorical(stacked_logits_flat)
        # sample_flat_ = cate_.sample([n_samples * n])
        # sample_shape_ = np.concatenate([[n, n_samples], self.batch_shape], axis=0).tolist()
        # sample_ = fluid.layers.reduce_sum(paddle.reshape(sample_flat_, sample_shape_), dim=0)
        # sample_ = paddle.cast(sample_, self.dtype)

        # TODO: Output shape now is [n_experiments, n_samples, batch_shape ],
        #  check here to see if the output shape should be [n_samples, -1 ] or not.

        sample_.stop_gradient = False
        sample_ = paddle.cast(sample_, self.dtype)

        self.sample_cache = sample_
        if n_samples > 1:
            assert (sample_.shape[0] == n_samples)
        # Output shape: [ batch_shape..., n_samples]
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        # if len(sample.shape) > len(self._probs.shape):
        #     _probs = paddle.tile(self._probs, repeat_times=\
        #                 [sample.shape[0], *len(self._probs.shape)*[1]])
        # else:
        #     _probs = self._probs

        ## Log Prob
        # TODO: Paddle do not have binomial module. Here we use Scipy
        log_prob = paddle.to_tensor(stats.binom.logpmf(
            sample.numpy(), self.n_experiments, 1 / (1. + np.exp(-self._probs.numpy()))))
        log_prob = paddle.cast(log_prob, self.dtype)

        # # A different way to calculate log_prob:
        # n = paddle.cast(self.n_experiments, self.param_dtype)
        # sample = paddle.cast(sample, self.param_dtype)
        #
        # log_1_minus_p = -paddle.nn.functional.softplus(_probs)
        # # TODO: Paddle do not have lgamma module. Here we use Math package
        # lgamma_n_plus_1 = paddle.to_tensor(list(map(lambda x: math.lgamma(x), n + 1 )))
        # lgamma_given_plus_1 = paddle.to_tensor(list(map(lambda x: math.lgamma(x), sample + 1)))
        # lgamma_n_minus_given_plus_1 = paddle.to_tensor(list(map(lambda x: math.lgamma(x), n - sample + 1)))
        #
        # log_prob = lgamma_n_plus_1 - lgamma_n_minus_given_plus_1 \
        #            - lgamma_given_plus_1 + sample * _probs + n * log_1_minus_p

        return log_prob


