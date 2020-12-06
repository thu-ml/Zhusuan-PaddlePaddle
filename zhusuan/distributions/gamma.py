import numpy as np
import math
import paddle
import paddle.fluid as fluid

from .base import Distribution

__all__ = [
    'Gamma',
]


class Gamma(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(Gamma, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._alpha = kwargs['alpha']
        self._beta = kwargs['beta']

    @property
    def alpha(self):
        """ The shape parameter of the Gamma distribution."""
        # alpha should not be 0
        return self._alpha

    @property
    def beta(self):
        """The inverse scale parameter of the Gamma distribution."""
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
        sample_ = paddle.cast(paddle.to_tensor(
            np.random.gamma(shape=_alpha,scale=_beta, size=sample_shape_)),
            dtype=self.dtype)

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_


    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        # alpha should not be 0
        _alpha, _beta = self.alpha, self.beta

        ## Log Prob
        log_given = paddle.log(sample)
        log_beta = paddle.log(_beta)
        # TODO: Paddle do not have lgamma module. Here we use Math package
        lgamma_alpha = paddle.to_tensor(list(map(lambda x: math.lgamma(x), _alpha )))
        log_prob = _alpha * log_beta - lgamma_alpha + (_alpha - 1) * log_given - \
            _beta * sample
        return log_prob

