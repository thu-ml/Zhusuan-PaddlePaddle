import numpy as np
import math
import paddle
import paddle.fluid as fluid

from .base import Distribution

__all__ = [
    'Poisson',
]


class Poisson(Distribution):
    def __init__(self,
                 dtype='int32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(Poisson, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._rate = kwargs['rate']

    @property
    def rate(self):
        """The rate parameter of Poisson."""
        return self._rate

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        raise self.rate.shape

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        return self.rate.shape

    def _sample(self, n_samples=1, **kwargs):

        _rate = self.rate
        sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()

        # TODO: Paddle do not have poisson distribution module. Here we use Numpy Random
        sample_ = paddle.cast(paddle.to_tensor(
            np.random.poisson(lam=_rate, size=sample_shape_)),
            dtype=self.dtype)

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_



    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        _rate = self.rate

        ## Log Prob
        log_rate = paddle.log(_rate)
        # TODO: Paddle do not have lgamma module. Here we use Math package
        lgamma_given_plus_1 = paddle.to_tensor(list(map(lambda x: math.lgamma(x), sample + 1 )))
        log_prob =  sample * log_rate - _rate - lgamma_given_plus_1

        return log_prob

