import numpy as np

from .base import Distribution

import paddle
import paddle.nn.functional as F

__all__ = [
    'Logistic',
]

class Logistic(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=False,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(Logistic, self).__init__(dtype,
                                       param_dtype,
                                       is_continues,
                                       is_reparameterized,
                                       group_ndims=group_ndims,
                                       **kwargs)
        self._loc = kwargs['loc']
        self._scale = kwargs['scale']
        if not isinstance(self._loc, paddle.Tensor):
            self._loc = paddle.to_tensor(self._loc)
        if not isinstance(self._scale, paddle.Tensor):
            self._scale = paddle.to_tensor(self._scale)

    def _sample(self, n_samples=1, **kwargs):
        if 'shape' in kwargs.keys():
            shape = kwargs['shape']
            uniform = np.random.uniform(low=0., high=1., size=shape)
            uniform = paddle.to_tensor(uniform)
            # uniform = paddle.distribution.Uniform(low=float(np.finfo(np.float32).tiny), high=1.).sample(shape=shape)
            sample_ = paddle.log(uniform) - paddle.log1p(-uniform)
            return sample_ * self._scale + self._loc
        else:
            return 0.

    def _log_prob(self, sample=None):
        if sample is None:
            raise NotImplementedError()
        z = (sample - self._loc) / self._scale
        return -z - 2. * F.softplus(-z) - paddle.log(self._scale)

