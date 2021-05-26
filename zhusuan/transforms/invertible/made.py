"""
Invertible NN transforms
"""

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform, Assign

from zhusuan.transforms.invertible import InvertibleTransform


class MaskedLinear(nn.Linear):
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        stdv = 1. / math.sqrt(n_outputs)
        super().__init__(input_size, n_outputs, weight_attr=Uniform(-stdv, stdv),
                         bias_attr=Uniform(-stdv, stdv))

        self.register_buffer('mask', mask)
        self.cond_label_size = cond_label_size

        if cond_label_size is not None:
            self.cond_weight = self.create_parameter(shape=[cond_label_size, n_outputs],
                                                     default_initializer=Assign(
                                                         paddle.rand([cond_label_size, n_outputs]) / math.sqrt(
                                                             cond_label_size)))
            self.add_parameter('cond_weight', self.cond_weight)

    def forward(self, x, cond_y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if cond_y is not None:
            out = out + F.linear(cond_y, self.cond_weight)
        return out


class MADE(InvertibleTransform):
    @staticmethod
    def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
        """
        Mask generator for MADE & MAF (see MADE paper sec 4:https://arxiv.org/abs/1502.03509)
        Args:
            input_size:
            hidden_size:
            n_hidden:
            input_order:
            input_degrees:

        Returns: List of masks

        """
        degrees = []

        if input_order == 'sequential':
            degrees += [paddle.arange(input_size)] if input_degrees is None else [input_degrees]
            for _ in range(n_hidden + 1):
                degrees += [paddle.arange(hidden_size) % (input_size - 1)]
            degrees += [paddle.arange(input_size) % input_size - 1] if input_degrees is None else [
                input_degrees % input_size - 1]

        elif input_order == 'random':
            # TODO: Implement random input order
            raise NotImplementedError()

        else:
            raise NotImplementedError()

        # construct masks
        masks = []
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [paddle.cast(d1.unsqueeze(0) >= d0.unsqueeze(-1), dtype='float32')]

        return masks, degrees[0]

    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None,
                 input_order='sequential', input_degrees=None):
        super(MADE, self).__init__()

        # create masks
        masks, self.input_degrees = self.create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net.extend([nn.ReLU(), MaskedLinear(hidden_size, hidden_size, m)])
        self.net.extend(
            [nn.ReLU(), MaskedLinear(hidden_size, 2 * input_size, paddle.tile(masks[-1], repeat_times=[1, 2]))])
        self.net = nn.Sequential(*self.net)

    def _forward(self, x, cond_y=None, **kwargs):
        m, loga = self.net(self.net_input(x, cond_y)).chunk(2, axis=1)
        u = (x - m) * paddle.exp(-loga)
        log_det = -loga
        return (u,), log_det

    def _inverse(self, u, cond_y=None, **kwargs):
        x = paddle.zeros_like(u)
        for i in self.input_degrees:
            i = int(i.numpy())
            m, loga = self.net(self.net_input(x, cond_y)).chunk(2, axis=1)
            x[:, i] = u[:, i] * paddle.exp(loga[:, i]) + m[:, i]
        log_det = loga
        return (x,), log_det
