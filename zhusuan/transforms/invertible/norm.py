import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform, Assign, Constant

from zhusuan.transforms.invertible import InvertibleTransform


class BatchNorm(InvertibleTransform):
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = self.create_parameter(shape=[input_size],
                                               default_initializer=Constant(0.))
        self.beta = self.create_parameter(shape=[input_size],
                                          default_initializer=Constant(0.))
        self.add_parameter('log_gamma', self.log_gamma)
        self.add_parameter('beta', self.beta)

        self.register_buffer('running_mean', paddle.zeros([input_size]))
        self.register_buffer('running_var', paddle.zeros([input_size]))

    def _forward(self, x, **kwargs):
        if self.training:
            self.batch_mean = x.mean(axis=0)
            self.batch_var = x.var(axis=0)

            self.running_mean = self.running_mean * self.momentum + self.batch_mean * (1 - self.momentum)
            self.running_var = self.running_var * self.momentum + self.batch_var * (1 - self.momentum)

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / paddle.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_det = self.log_gamma - 0.5 * paddle.log(var + self.eps)
        return (y,), paddle.expand_as(log_det, x)

    def _inverse(self, y, **kwargs):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (y - self.beta) * paddle.exp(-self.log_gamma)
        x = x_hat * paddle.sqrt(var + self.eps) + mean
        log_det = 0.5 * paddle.log(var + self.eps) - self.log_gamma
        return (x,), paddle.expand_as(log_det, x)


class ActNormBase(InvertibleTransform):
    def __init__(self, eps=1e-6):
        super(ActNormBase, self).__init__()
        self.eps = eps
        self.register_buffer('inited', paddle.to_tensor(0, dtype='uint8'))

    def init_parameters(self, x):
        # Data dependent init, per channel mean and variance
        raise NotImplementedError()

    def compute_log_det(self, x):
        raise NotImplementedError()

    def _forward(self, x, **kwargs):
        if self.inited.numpy() == 0:
            self.init_parameters(x)
            self.inited += 1
        y = (x - self.shift) * paddle.exp(-self.log_scale)
        log_det = self.compute_log_det(x)
        return y, log_det

    def _inverse(self, y, **kwargs):
        x = y * paddle.exp(self.log_scale) + self.shift
        log_det = -self.compute_log_det(y)
        return x, log_det

class ActNorm(ActNormBase):
    """
    Activation Normalization for shape [B, D]
    """
    def __init__(self):
        super(ActNorm, self).__init__()

    def init_parameters(self, x):
        with paddle.no_grad():
            self.shift = self.create_parameter(shape=[1, x.shape[1]],
                                               default_initializer=Assign(paddle.mean(x, axis=0, keepdim=True)))
            self.log_scale = self.create_parameter(shape=[1, x.shape[1]],
                                                   default_initializer=Assign(paddle.log(
                                                       paddle.std(x, axis=0, keepdim=True) + self.eps)))
            self.add_parameter('shift', self.shift)
            self.add_parameter('log_scale', self.log_scale)

    def compute_log_det(self, x):
        return paddle.sum(-self.log_scale).expand(shape=[x.shape[0]])

