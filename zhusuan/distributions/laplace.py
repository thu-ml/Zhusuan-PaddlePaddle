import numpy as np
import paddle
import paddle.fluid as fluid

import numpy as np
import math

from .base import Distribution

__all__ = [
    'Laplace',
]


class Laplace(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(Laplace, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._loc = kwargs['loc']
        self._scale = kwargs['scale']

    @property
    def loc(self):
        """The location parameter of the Laplace distribution."""
        return self._loc

    @property
    def scale(self):
        """The scale parameter of the Laplace distribution."""
        return self._scale

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        raise (self._loc + self._scale).shape

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return (self._loc + self._scale).shape

    def _sample(self, n_samples=1, **kwargs):

        # samples must be sampled from (-1, 1) rather than [-1, 1)
        _loc, _scale = self.loc, self.scale
        if not self.is_reparameterized:
            _loc.stop_gradient = True
            _scale.stop_gradient = True
        sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        uniform_sample_ = paddle.cast(paddle.uniform( shape=sample_shape_,
                                                      min=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                                                                       self.dtype.as_numpy_dtype(0.)),
                                                      max=1. ), dtype=self.dtype)
        sample_ = _loc - _scale * paddle.sign(uniform_sample_) * \
            paddle.log1p(-paddle.abs(uniform_sample_))

        self.sample_cache = sample_
        assert (sample_.shape[0] == n_samples)
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        _loc, _scale = self.loc, self.scale

        ## Log Prob
        log_scale = paddle.log(_scale)
        log_prob = -np.log(2.) - log_scale - paddle.abs(sample - _loc) / _scale

        return log_prob
