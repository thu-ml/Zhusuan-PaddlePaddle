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

        if type(kwargs['minval']) in [type(1.), type(1)]:
            self._minval = paddle.cast(paddle.to_tensor([kwargs['minval']]), self.dtype)
        else:
            self._minval = paddle.reshape(kwargs['minval'], [1]) if kwargs['minval'].ndim == 0 \
                else kwargs['minval']
        if type(kwargs['maxval']) in [type(1.), type(1)]:
            self._maxval = paddle.cast(paddle.to_tensor([kwargs['maxval']]), self.dtype)
        else:
            self._maxval = paddle.reshape(kwargs['maxval'], [1])  if kwargs['maxval'].ndim == 0 \
                else kwargs['maxval']

        self._dtype = dtype if type(dtype) in [ str ] and not dtype is 'float32' else self._minval.dtype

        # self._minval = kwargs['minval']
        # self._maxval = kwargs['maxval']

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

        sample_ = paddle.cast(sample_, self.dtype)
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
        _minval *= paddle.ones(self.batch_shape, dtype=self.param_dtype)
        _maxval *= paddle.ones(self.batch_shape, dtype=self.param_dtype)

        ## Log Prob
        sample = paddle.cast(sample, self.param_dtype)
        _minval = paddle.cast(_minval, self.param_dtype)
        _maxval = paddle.cast(_maxval, self.param_dtype)

        lb_bool = _minval <= sample*paddle.ones_like(_minval)#paddle.less_than(_minval, sample)
        ub_bool = sample*paddle.ones_like(_maxval) < _maxval
        mask = paddle.cast( paddle.logical_and(lb_bool, ub_bool), self.dtype)
        p = 1. / (_maxval - _minval)
        prob_ = p * mask
        log_prob = paddle.log( prob_ )
        log_prob = paddle.cast(log_prob, self.dtype)
        return log_prob

        # # A different way to calculate log_prob:
        # mask = paddle.cast(  paddle.logical_and( paddle.less_equal(_minval, sample),
        #                                          paddle.less_than(sample, _maxval) ),
        #                      self.dtype )
        # p = 1. / (_maxval - _minval)
        # prob_ = p * mask
        # log_prob = paddle.log( prob_ )
        #
        # return log_prob




