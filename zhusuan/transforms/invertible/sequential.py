from __future__ import absolute_import
from __future__ import division

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
import numpy as np
import scipy

from zhusuan.transforms.invertible.base import InvertibleTransform

__all__ = [
    'Sequential',
]


class Sequential(InvertibleTransform):
    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = paddle.nn.LayerList(layers)

    def _forward(self, *x, **kwargs):
        logdet_terms = []
        for i in range(len(self.layers)):
            x = self.layers[i](*x, reverse=False, **kwargs)
            assert isinstance(x, tuple)
            assert len(x) >= 2
            if x[-1] is not None:
                logdet_terms.append(x[-1])
            if isinstance(x[0], tuple):
                x = x[0]
            else:
                x = x[:len(x) - 1]
            assert isinstance(x, tuple)
        return x, sum(logdet_terms) if logdet_terms else paddle.zeros([])

    def _inverse(self, *y, **kwargs):
        logdet_terms = []
        for i in reversed(range(len(self.layers))):
            y = self.layers[i](*y, reverse=True, **kwargs)
            assert isinstance(y, tuple)
            assert len(y) >= 2
            if y[-1] is not None:
                logdet_terms.append(y[-1])
            if isinstance(y[0], tuple):
                y = y[0]
            else:
                y = y[:len(y) - 1]
            assert isinstance(y, tuple)
        return y, sum(logdet_terms) if logdet_terms else paddle.zeros([])
