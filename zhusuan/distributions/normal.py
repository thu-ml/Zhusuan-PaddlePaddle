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
                 **kwargs):
        super(Normal, self).__init__(dtype, 
                             param_dtype, 
                             is_continues,
                             is_reparameterized)
        self.std = kwargs['std']
        self.mean = kwargs['mean']


    def sample(self, **kwargs):
        if self.is_reparameterized:
            epsilon = paddle.normal(name='sample',
                                    shape=fluid.layers.shape(self.std),
                                    mean=0.0,
                                    std=1.0)
            sample_ = self.mean + self.std * epsilon
        else:
            sample_ = paddle.normal(name='sample',
                                   shape=fluid.layers.shape(self.std),
                                   mean=self.mean,
                                   std=self.std)
        self.sample_cache = sample_

        #self.nodes[name] = (sample, log_prob)

        #assert([sample.shape[0]] == log_prob.shape)
        return sample_

    def log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        ## Log Prob
        logstd = paddle.log(self.std)
        c = -0.5 * np.log(2 * np.pi)
        precision = paddle.exp(-2 * logstd)
        log_prob_sample = c - logstd - 0.5 * precision * paddle.square(sample - self.mean)
        log_prob = fluid.layers.reduce_sum(log_prob_sample, dim=1)

        return log_prob



