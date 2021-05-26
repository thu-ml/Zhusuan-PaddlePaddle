import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform, Assign, Constant

import numpy as np
import scipy.linalg

from zhusuan.transforms.invertible import InvertibleTransform


class InvertibleConv1x1(InvertibleTransform):
    def __init__(self, num_channels):
        super(InvertibleConv1x1, self).__init__()
        self.num_channels = num_channels
        rot_mat = np.linalg.qr(np.random.randn(num_channels, num_channels))[0].astype('float32')
        p, l, u = scipy.linalg.lu(rot_mat)
        s = np.diag(u)
        u = np.triu(u, 1)

        u_mask = np.triu(np.ones_like(u), 1)
        l_mask = u_mask.T

        p = paddle.to_tensor(p)
        l = paddle.to_tensor(l)
        s = paddle.to_tensor(s)
        u = paddle.to_tensor(u)

        self.register_buffer('P', p)
        self.register_buffer('U_mask', paddle.to_tensor(u_mask))
        self.register_buffer('L_mask', paddle.to_tensor(l_mask))
        self.register_buffer('s_sign', paddle.sign(s))
        self.register_buffer('L_eye', paddle.eye(l_mask.shape[0]))

        self.L = self.create_parameter(shape=l.shape,
                                       default_initializer=Assign(l))
        self.s = self.create_parameter(shape=s.shape,
                                       default_initializer=Assign(paddle.log(paddle.abs(s))))
        self.U = self.create_parameter(shape=u.shape,
                                       default_initializer=Assign(u))
        self.add_parameter('L', self.L)
        self.add_parameter('s', self.s)
        self.add_parameter('U', self.U)

    @property
    def W(self):
        weight = self.P. \
                     matmul(self.L * self.L_mask + self.L_eye). \
                     matmul(self.U * self.U_mask) + paddle.diag(self.s_sign * paddle.exp(self.s))
        return weight

    def _conv(self, h, inverse=False):
        batch, channel, *feature = h.shape
        len_features = len(feature)
        weight = self.W
        if inverse:
            weight = paddle.inverse(weight)
        for _ in range(len_features):
            weight = weight.unsqueeze(-1)

        if len_features == 1:
            return F.conv1d(h, weight)
        elif len_features == 2:
            return F.conv2d(h, weight)
        elif len_features == 3:
            return F.conv3d(h, weight)
        else:
            return NotImplementedError()

    def _ldj_multiplier(self, h):
        b, c, *feature_dim = h.shape
        return paddle.numel(paddle.ones(shape=feature_dim))

    def _forward(self, x, **kwargs):
        y = self._conv(x)
        log_det = paddle.sum(self.s) * self._ldj_multiplier(x)
        return y, log_det

    def _inverse(self, y, **kwargs):
        x = self._conv(y, inverse=True)
        return x, None

