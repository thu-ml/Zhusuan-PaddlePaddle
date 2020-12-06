
import paddle
import paddle.fluid as fluid

import numpy as np
import scipy as sp
import math

from .base import Distribution

__all__ = [
    'MatrixVariateNormalCholesky',
]


class MatrixVariateNormalCholesky(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(MatrixVariateNormalCholesky, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._mean = kwargs['mean']
        self._n_row = self._mean.shape[-2]
        self._n_col = self._mean.shape[-1]
        self._u_tril = kwargs['u_tril']
        self._v_tril = kwargs['v_tril']

    @property
    def mean(self):
        """The mean of the matrix variate normal distribution."""
        return self._mean

    @property
    def u_tril(self):
        """
        The lower triangular matrix in the Cholesky decomposition of the
        among-row covariance.
        """
        return self._u_tril

    @property
    def v_tril(self):
        """
        The lower triangular matrix in the Cholesky decomposition of the
        among-column covariance.
        """
        return self._v_tril


    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        raise self._mean.shape[:-2]

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return self._mean.shape[:-2]

    def _sample(self, n_samples=1, **kwargs):

        _mean, _u_tril, _v_tril = self.mean, self.u_tril, self.v_tril
        if not self.is_reparameterized:
            _mean.stop_gradient = True
            _u_tril.stop_gradient = True
            _v_tril.stop_gradient = True

        _len = len(_mean.shape)
        batch_u_tril = paddle.tile(_u_tril, repeat_times=[n_samples, *_len*[1]])
        batch_v_tril = paddle.tile(_v_tril, repeat_times=[n_samples, *_len*[1]])

        sample_shape = np.concatenate([[n_samples], _mean.shape ], axis=0).tolist()
        noise = paddle.cast( paddle.normal(name='sample',
                                           shape=sample_shape), dtype= self.dtype )

        import tensorflow.compat.v1 as tf
        transpose_orders = [ i for i in range(len(batch_v_tril.shape)-2) ]
        transpose_orders.extend([len(batch_v_tril.shape)-1, len(batch_v_tril.shape)-2])
        sample_ = _mean + paddle.matmul(paddle.matmul(batch_u_tril, noise),
                                        paddle.transpose(batch_v_tril, transpose_orders))

        self.sample_cache = sample_
        assert(sample_.shape[0] == n_samples)
        return sample_


    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        _mean, _u_tril, _v_tril = self.mean, self.u_tril, self.v_tril

        ## Log Prob
        # TODO: Paddle do not have matrix_diag_part func,
        #  Should check if the diag func in Paddle match the requirements.
        # u_tril_shape = _u_tril.shape[0:]
        u_diag_batch = paddle.reshape(paddle.concat([paddle.diag(_u_tril[i])
                                                   for i in range(_u_tril.shape[0])], axis=0),
                                    shape=_u_tril.shape[:-1])
        log_det_u = 2 * fluid.layers.reduce_sum( paddle.log( u_diag_batch ), dim=-1)
        v_diag_batch = paddle.reshape(paddle.concat([paddle.diag(_v_tril[i])
                                                   for i in range(_v_tril.shape[0])], axis=0),
                                    shape=_v_tril.shape[:-1])
        log_det_v = 2 * fluid.layers.reduce_sum( paddle.log(v_diag_batch), dim=-1)
        _n_row = paddle.cast(self._n_row, self.dtype)
        _n_col = paddle.cast(self._n_col, self.dtype)
        logZ = - (_n_row * _n_col) / 2. * \
               paddle.log(2. * paddle.cast(paddle.to_tensor(np.pi), dtype=self.dtype)) - \
               _n_row / 2. * log_det_v - _n_col / 2. * log_det_u
        # logZ.shape == batch_shape
        y = sample - _mean
        ones_temp = paddle.ones(y.shape[:-1])

        expand_dim_shape = np.concatenate([ones_temp.shape, [1]], axis=0).tolist()
        y_with_last_dim_changed = paddle.reshape(ones_temp, expand_dim_shape)

        # TODO: Better to add some broadcast check functions here.
        Lu = _u_tril
        # Lu, _ = maybe_explicit_broadcast(
        #     _u_tril, y_with_last_dim_changed,
        #     'MatrixVariateNormalCholesky.u_tril', 'expand_dims(given, -1)')

        shape_temp = np.concatenate([y.shape[:-2], y.shape[-1:]], axis=0).tolist()
        ones_temp = paddle.ones( shape_temp )
        expand_dim_shape = np.concatenate([ones_temp.shape, [1]], axis=0).tolist()
        y_with_sec_last_dim_changed = paddle.reshape(ones_temp, expand_dim_shape)

        # TODO: Better to add some broadcast check functions here.
        Lv = _v_tril
        # Lv, _ = maybe_explicit_broadcast(
        #     _v_tril, y_with_sec_last_dim_changed,
        #     'MatrixVariateNormalCholesky.v_tril',
        #     'expand_dims(given, -1)')

        # TODO: Paddle do not have matrix_triangular_solve func, Here we use Scipy package.
        x_Lb_inv_t = paddle.to_tensor(sp.linalg.solve_triangular(Lu.numpy(), y.numpy(), lower=True))

        transpose_orders = [ i for i in range(len(x_Lb_inv_t.shape)-2) ]
        transpose_orders.extend([len(x_Lb_inv_t.shape)-1, len(x_Lb_inv_t.shape)-2])
        x_t = paddle.to_tensor(sp.linalg.solve_triangular(
            Lv.numpy(), paddle.transpose(x_Lb_inv_t, transpose_orders).numpy(), lower=True))

        stoc_dist = -0.5 * fluid.layers.reduce_sum(paddle.square(x_t), dim=[-1, -2])
        log_prob = logZ + stoc_dist

        return log_prob


