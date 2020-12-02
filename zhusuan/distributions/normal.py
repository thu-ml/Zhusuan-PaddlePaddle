import numpy as np
import paddle
import paddle.fluid as fluid

from .base import Distribution

class Normal(Distribution):
    def __init__(self,
                 dtype='float32',
                 param_dtype='float32',
                 is_continues=True,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(Normal, self).__init__(dtype, 
                             param_dtype, 
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)
        try:
            self._std = kwargs['std']
        except:
            self._logstd = kwargs['logstd']
            self._std = paddle.exp(kwargs['logstd'])

        self._mean = kwargs['mean']

    @property
    def mean(self):
        """The mean of the Normal distribution."""
        return self._mean

    @property
    def logstd(self):
        """The log standard deviation of the Normal distribution."""
        try:
            return self._logstd
        except:
            self._logstd = paddle.log(self._std)
            return self._logstd

    @property
    def std(self):
        """The standard deviation of the Normal distribution."""
        return self._std

    def _sample(self, n_samples=1, **kwargs):
        #print('n_samples: ', n_samples)
        _shape = fluid.layers.shape(self._mean)
        _shape = fluid.layers.concat([paddle.to_tensor([n_samples], dtype="int32"), _shape])
        _len = len(self._std.shape)
        _std = paddle.tile(self._std, repeat_times=[n_samples, *_len*[1]]) 
        _mean = paddle.tile(self._mean, repeat_times=[n_samples, *_len*[1]]) 
        #print('_shape: ', _shape)

        if self.is_reparameterized:
            #print('_std: ', _std)
            epsilon = paddle.normal(name='sample',
                                    shape=_shape,
                                    mean=0.0,
                                    std=1.0)
            sample_ = _mean + _std * epsilon
        else:
            sample_ = paddle.normal(name='sample',
                                   shape=_shape,
                                   mean=_mean,
                                   std=_std)
        self.sample_cache = sample_
        #print('sample_.shape: ', sample_.shape)

        assert(sample_.shape[0] == n_samples)
        return sample_

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        if len(sample.shape) > len(self._std.shape):
            n_samples = sample.shape[0]
            _len = len(self._std.shape)
            _std = paddle.tile(self._std, repeat_times=[n_samples, *_len*[1]]) 
            _mean = paddle.tile(self._mean, repeat_times=[n_samples, *_len*[1]]) 
        else:
            _std = self._std
            _mean = self._mean

        ## Log Prob
        logstd = paddle.log(_std)
        c = -0.5 * np.log(2 * np.pi)
        precision = paddle.exp(-2 * logstd)
        log_prob_sample = c - logstd - 0.5 * precision * paddle.square(sample - _mean)
        log_prob = fluid.layers.reduce_sum(log_prob_sample, dim=-1)

        return log_prob
