import numpy as np
import math
import paddle
import paddle.fluid as fluid
from scipy import stats

from .base import Distribution

__all__ = [
    'InverseGamma',
]


class InverseGamma(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(InverseGamma, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._alpha = kwargs['alpha']
        self._beta = kwargs['beta']

    @property
    def alpha(self):
        """ The shape parameter of the InverseGamma distribution."""
        # alpha should not be 0
        return self._alpha

    @property
    def beta(self):
        """The scale parameter of the InverseGamma distribution."""
        return self._beta

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return (self._alpha + self._beta).shape

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return (self._alpha + self._beta).shape

    def _sample(self, n_samples=1, **kwargs):

        _alpha, _beta = self.alpha, self.beta

        # Broadcast
        _alpha *= paddle.ones(self.batch_shape, dtype=_alpha.dtype)
        _beta *= paddle.ones(self.batch_shape, dtype=_beta.dtype)

        if not self.is_reparameterized:
            _alpha, _beta = _alpha * 1, _beta * 1
            _alpha.stop_gradient = True
            _beta.stop_gradient = True

        if n_samples > 1:
            sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        else:
            sample_shape_ = self.batch_shape
        # TODO: Paddle do not have gamma distribution module. Here we use Numpy Random
        sample_gamma_ = paddle.cast(paddle.to_tensor(
            np.random.gamma(shape=_alpha.numpy(),scale=_beta.numpy(), size=sample_shape_)),
            dtype=self.dtype)
        sample_ = 1/sample_gamma_

        sample_ = paddle.cast(sample_, self._alpha.dtype)
        sample_.stop_gradient = False

        self.sample_cache = sample_
        if n_samples > 1:
            assert(sample_.shape[0] == n_samples)
        return sample_


    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        # alpha should not be 0
        _alpha, _beta = self.alpha, self.beta

        # # Broadcast
        # _alpha *= paddle.ones(self.batch_shape, dtype=_alpha.dtype)
        # _beta *= paddle.ones(self.batch_shape, dtype=_beta.dtype)

        ## Log Prob
        # TODO: Paddle do not have lgamma module. Here we use Scipy
        log_prob = paddle.to_tensor(stats.invgamma.logpdf(
            sample.numpy(), _alpha.numpy(), scale= _beta.numpy()))
        log_prob = paddle.cast(log_prob, self._alpha.dtype)

        # # A different way to calculate log_prob:
        # log_given = paddle.log(sample)
        # log_beta = paddle.log(_beta)
        # lgamma_alpha = paddle.to_tensor(list(map(lambda x: math.lgamma(x), _alpha )))
        # log_prob = _alpha * log_beta - lgamma_alpha - (_alpha + 1) * log_given - _beta / sample

        return log_prob
