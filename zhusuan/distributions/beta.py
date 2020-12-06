import numpy as np
import math
import paddle
import paddle.fluid as fluid

from .base import Distribution

__all__ = [
    'Beta',
]


class Beta(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(Beta, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._alpha = kwargs['alpha']
        self._beta = kwargs['beta']

    @property
    def alpha(self):
        """One of the two shape parameters of the Beta distribution."""
        return self._alpha

    @property
    def beta(self):
        """One of the two shape parameters of the Beta distribution."""
        return self._beta

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        raise (self._alpha + self._beta).shape

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return (self._alpha + self._beta).shape

    def _sample(self, n_samples=1, **kwargs):

        _alpha, _beta = self.alpha, self.beta
        sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        # TODO: Paddle do not have gamma distribution module. Here we use Numpy Random
        x_ = paddle.cast(paddle.to_tensor(
            np.random.gamma(shape=_alpha,scale=1, size=sample_shape_)),
            dtype=self.dtype)
        y_ = paddle.cast(paddle.to_tensor(
            np.random.gamma(shape=_beta,scale=1, size=sample_shape_)),
            dtype=self.dtype)
        sample_ = x_ / (x_ + y_)

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_


    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        # alpha and beta should not be 0
        _alpha, _beta = self.alpha, self.beta

        ## Log Prob
        log_given = paddle.log(sample)
        log_1_minus_given = paddle.log(1 - sample)
        # TODO: Paddle do not have lgamma module. Here we use Math package
        lgamma_alpha, lgamma_beta = paddle.to_tensor(list(map(lambda x: math.lgamma(x), _alpha))),\
                                    paddle.to_tensor(list(map(lambda x: math.lgamma(x), _beta)))
        lgamma_alpha_plus_beta = paddle.to_tensor(list(map(lambda x: math.lgamma(x), _alpha + _beta )))

        log_prob = (_alpha - 1) * log_given + (_beta - 1) * log_1_minus_given - (
            lgamma_alpha + lgamma_beta - lgamma_alpha_plus_beta)

        return log_prob

