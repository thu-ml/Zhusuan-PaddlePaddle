import paddle
import numpy as np
import math

from .base import InvertibleTransform


def get_coupling_mask(n_dim, n_channel, n_mask, split_type='OddEven'):
    '''

    Args:
        n_dim:
        n_channel:
        n_mask:
        split_type:

    Returns: List of generated mask

    '''
    masks = []
    if split_type == 'OddEven':
        if n_channel == 1:
            mask = paddle.arange(n_dim, dtype='float32') % 2
            for i in range(n_mask):
                masks.append(mask)
                mask = 1. - mask
    # elif split_type == 'Half':
    #     pass
    # elif split_type == 'RandomHalf':
    #     pass
    # else:
    #     raise NotImplementedError()
    return masks


class AdditiveCoupling(InvertibleTransform):
    def __init__(self, in_out_dim=-1, mid_dim=-1, hidden=-1, mask=None, inner_nn=None):
        super().__init__()
        if inner_nn is None:
            # default inner nn, refer to NICE
            stdv = 1. / math.sqrt(mid_dim)
            stdv2 = 1. / math.sqrt(in_out_dim)

            self.nn = []
            self.nn += [paddle.nn.Linear(in_out_dim, mid_dim, weight_attr=paddle.nn.initializer.Uniform(-stdv, stdv),
                                         bias_attr=paddle.nn.initializer.Uniform(-stdv, stdv)),
                        paddle.nn.ReLU()]
            for _ in range(hidden - 1):
                self.nn += [paddle.nn.Linear(mid_dim, mid_dim, weight_attr=paddle.nn.initializer.Uniform(-stdv, stdv),
                                             bias_attr=paddle.nn.initializer.Uniform(-stdv, stdv)),
                            paddle.nn.ReLU()]
            self.nn.append(paddle.nn.Linear(mid_dim, in_out_dim,
                                            weight_attr=paddle.nn.initializer.Uniform(-stdv2, stdv2),
                                            bias_attr=paddle.nn.initializer.Uniform(-stdv2, stdv2)))
            self.nn = paddle.nn.Sequential(*self.nn)

        else:
            self.nn = inner_nn

        self.mask = mask

    def _forward(self, x, **kwargs):
        x1, x2 = self.mask * x, (1. - self.mask) * x
        shift = self.nn(x1)
        y1, y2 = x1, x2 + shift * (1. - self.mask)
        return y1 + y2, None

    def _inverse(self, y, **kwargs):
        y1, y2 = self.mask * y, (1. - self.mask) * y
        shift = self.nn(y1)
        x1, x2 = y1, y2 - shift * (1. - self.mask)
        return x1 + x2, None
