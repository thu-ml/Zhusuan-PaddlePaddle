import paddle
import paddle.fluid as fluid
import numpy as np
from scipy import stats

from .base import Distribution


__all__ = [
    'Bernoulli',
]

class Bernoulli(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=False,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(Bernoulli, self).__init__(dtype, 
                             param_dtype, 
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)
        self._probs = kwargs['probs']
        # self._dtype = self._probs.dtype #if self._probs.dtype != paddle.zeros([1], dtype=dtype)
        # if (not type(dtype) is str) or dtype != 'float32':
        #     self._dtype = dtype
    @property
    def probs(self):
        """The odds of probabilities of being 1."""
        return self._probs

    def _batch_shape(self):
        return self.probs.shape

    def _sample(self, n_samples=1, **kwargs):
        if n_samples > 1:
            sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
            _probs = self._probs * paddle.ones(sample_shape_)
        else:
            _probs = self._probs

        # _probs = paddle.cast(_probs, self.param_dtype)
        _probs *= paddle.cast(_probs <= 1, self.param_dtype )
        sample_ = paddle.bernoulli(_probs)
        sample_ = paddle.cast(sample_, dtype=self.dtype)

        self.sample_cache = sample_
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        ## Log Prob
        if len(sample.shape) > len(self._probs.shape):
            sample_shape_ = np.concatenate([[sample.shape[0]], self.batch_shape], axis=0).tolist()
            _probs = self._probs * paddle.ones(sample_shape_, dtype=self.param_dtype)
        else:
            _probs = self._probs

        # add 1e-8 for numerical stable
        _probs = _probs + 1e-8
        log_prob = sample * paddle.log( _probs ) \
                            + (1 - sample) * paddle.log(1 - _probs )
        log_prob = paddle.cast(log_prob, dtype=self.dtype)

        # log_prob = fluid.layers.reduce_sum(log_prob, dim=-1)

        return log_prob



