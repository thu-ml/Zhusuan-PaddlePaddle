
import paddle
import paddle.fluid as fluid

import numpy as np
import scipy as sp
import math

from .base import Distribution
from .utils import log_combination

__all__ = [
    'Dirichlet'
]


class Dirichlet(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(Dirichlet, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._alpha = kwargs['alpha']
        self._n_categories = self._alpha.shape[-1]

    @property
    def alpha(self):
        """The concentration parameter of the Dirichlet distribution."""
        return self._alpha

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        raise self._alpha.shape[:-1]

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return self._alpha.shape[:-1]

    def _sample(self, n_samples=1, **kwargs):

        _alpha = self.alpha

        sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        # TODO: Paddle do not have gamma distribution module. Here we use Numpy Random
        sample_ = paddle.cast(paddle.to_tensor(
            np.random.gamma(shape=_alpha, scale=1, size=sample_shape_)),
            dtype=self.dtype)
        sample_ = sample_ / fluid.layers.reduce_sum(sample_, dim=-1, keep_dim=True)

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_


    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache
        _alpha = self.alpha

        ## Log Prob

        # TODO: Paddle do not have lbeta func. Should find a way to achieve lbeta process.
        #  Calculation results here might not be correct.
        beta_sp = sp.stats.beta(_alpha, 1)
        lbeta_alpha = fluid.layers.reduce_sum(paddle.log(paddle.abs( paddle.to_tensor(beta_sp.pdf(_alpha)) )), dim=-1)

        # fix of no static shape inference for tf.lbeta
        log_sample = paddle.log(sample)
        log_prob = - lbeta_alpha + fluid.layers.reduce_sum((_alpha - 1) * log_sample, dim=-1)

        return log_prob



