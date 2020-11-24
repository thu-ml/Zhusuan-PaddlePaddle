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


    def _sample(self, **kwargs):
        if self.is_reparameterized:
            epsilon = paddle.normal(name='sample',
                                    shape=fluid.layers.shape(self._std),
                                    mean=0.0,
                                    std=1.0)
            sample_ = self._mean + self._std * epsilon
        else:
            sample_ = paddle.normal(name='sample',
                                   shape=fluid.layers.shape(self._std),
                                   mean=self._mean,
                                   std=self._std)
        self.sample_cache = sample_
        #assert([sample.shape[0]] == log_prob.shape)
        return sample_

    def _log_prob(self, sample=None):
        if sample is None:
            sample = self.sample_cache

        ## Log Prob
        logstd = paddle.log(self._std)
        c = -0.5 * np.log(2 * np.pi)
        precision = paddle.exp(-2 * logstd)
        log_prob_sample = c - logstd - 0.5 * precision * paddle.square(sample - self.mean)
        log_prob = fluid.layers.reduce_sum(log_prob_sample, dim=1)

        return log_prob
