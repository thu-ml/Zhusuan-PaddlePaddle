import numpy as np
import paddle
import paddle.fluid as fluid
from scipy import stats

from .base import Distribution

__all__ = [
    'Uniform',
]


class Uniform(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):

        super(Uniform, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._minval = kwargs['minval']
        self._maxval = kwargs['maxval']

    @property
    def minval(self):
        """ The lower bound on the range of the uniform distribution."""
        return self._minval

    @property
    def maxval(self):
        """The upper bound on the range of the uniform distribution."""
        return self._maxval

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.

        return (paddle.ones_like(self._minval) * paddle.ones_like(self._maxval)).shape
        # return (self._minval + self._maxval).shape

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        # PaddlePaddle will broadcast the tensor during the calculation.
        return (paddle.ones_like(self._minval) * paddle.ones_like(self._maxval)).shape
        # return (self._minval + self._maxval).shape

    def _sample(self, n_samples=1, **kwargs):

        _minval, _maxval = self.minval, self.maxval

        if not self.is_reparameterized:
            _minval, _maxval = _minval * 1, _maxval * 1
            _minval.stop_gradient = True
            _maxval.stop_gradient = True

        if n_samples > 1:
            sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        else:
            sample_shape_ = self.batch_shape
        sample_temp = paddle.cast(paddle.uniform( shape=sample_shape_, min=0, max=1 ),
                              dtype=self.dtype)
        sample_ = sample_temp * (_maxval - _minval) + _minval

        sample_ = paddle.cast(sample_, self._minval.dtype)
        sample_.stop_gradient = False
        self.sample_cache = sample_
        if n_samples > 1:
            assert(sample_.shape[0] == n_samples)
        return sample_

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        _minval, _maxval = self.minval, self.maxval

        # Broadcast
        _minval *= paddle.ones(self.batch_shape, dtype=_minval.dtype)
        _maxval *= paddle.ones(self.batch_shape, dtype=_maxval.dtype)

        ## Log Prob
        log_prob = paddle.to_tensor(stats.uniform.logpdf(
            sample.numpy(), _minval.numpy(), _maxval.numpy() - _minval.numpy()))
        log_prob = paddle.cast(log_prob, self._minval.dtype)

        # # A different way to calculate log_prob:
        # mask = paddle.cast(  paddle.logical_and( paddle.less_equal(_minval, sample),
        #                                          paddle.less_than(sample, _maxval) ),
        #                      self.dtype )
        # p = 1. / (_maxval - _minval)
        # prob_ = p * mask
        # log_prob = paddle.log( prob_ )

        return log_prob




