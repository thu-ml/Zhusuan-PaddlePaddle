
import paddle
import paddle.fluid as fluid

import numpy as np
import scipy as sp
import math

from .base import Distribution

__all__ = [
    'MultivariateNormalCholesky',
]


class MultivariateNormalCholesky(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(MultivariateNormalCholesky, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._mean = kwargs['mean']
        self._n_dim = self._mean.shape[-1]
        self._cov_tril = kwargs['cov_tril']

    @property
    def mean(self):
        """The mean of the normal distribution."""
        return self._mean

    @property
    def cov_tril(self):
        """
        The lower triangular matrix in the cholosky decomposition of the
        covariance.
        """
        return self._cov_tril

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        raise self._mean.shape[:-1]

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return self._mean.shape[:-1]

    def _sample(self, n_samples=1, **kwargs):

        _mean, _cov_tril = self.mean, self.cov_tril
        if not self.is_reparameterized:
            _mean.stop_gradient = True
            _cov_tril.stop_gradient = True

        _len = len(_mean.shape)
        batch_mean = paddle.tile(_mean, repeat_times=[n_samples, *_len*[1]])
        batch_cov = paddle.tile(_cov_tril, repeat_times=[n_samples, *_len * [1]])

        expand_dim_shape = np.concatenate([batch_mean.shape, [1] ], axis=0).tolist()
        # n_dim -> n_dim x 1 for matmul
        batch_mean = paddle.reshape(batch_mean, expand_dim_shape)
        noise = paddle.cast( paddle.normal(name='sample',
                                             shape=batch_mean.shape), dtype= self.dtype )
        sample_ = paddle.matmul(batch_cov, noise) + batch_mean
        sample_ = paddle.squeeze(sample_, axis=-1)

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        _mean, _cov_tril = self.mean, self.cov_tril

        ## Log Prob
        # TODO: Paddle do not have matrix_diag_part func,
        #  Should check if the diag func in Paddle match the requirements.
        # tril_shape = _cov_tril.shape[0:]
        diag_batch = paddle.reshape(paddle.concat([paddle.diag(_cov_tril[i])
                                                   for i in range(_cov_tril.shape[0])], axis=0),
                                    shape=_cov_tril.shape[:-1])
        log_det = 2 * fluid.layers.reduce_sum( paddle.log(diag_batch), axis=-1)
        _n_dim = paddle.cast(self._n_dim, self.dtype)
        log_z = - _n_dim / 2 * paddle.log(
            2 * paddle.cast(paddle.to_tensor(np.pi), dtype=self.dtype)) - log_det / 2
        # log_z.shape == batch_shape
        # (given-mean)' Sigma^{-1} (given-mean) =
        # (g-m)' L^{-T} L^{-1} (g-m) = |x|^2, where Lx = g-m =: y.
        y = sample - _mean
        expand_dim_shape = np.concatenate([y.shape, [1]], axis=0).tolist()
        y = paddle.reshape(y, expand_dim_shape)

        # TODO: Better to add some broadcast check functions here.
        # _cov_tril, _ = maybe_explicit_broadcast(
        #     cov_tril, y, 'MultivariateNormalCholesky.cov_tril',
        #     'expand_dims(given, -1)')

        # TODO: Paddle do not have matrix_triangular_solve func, Here we use Scipy package.
        x = paddle.to_tensor(sp.linalg.solve_triangular(_cov_tril.numpy(), y.numpy(), lower=True))
        x = paddle.squeeze(x, axis=-1)
        stoc_dist = -0.5 * fluid.layers.reduce_sum(paddle.square(x), dim=-1)
        log_prob = log_z + stoc_dist

        return log_prob

