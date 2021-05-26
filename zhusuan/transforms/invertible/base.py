from __future__ import absolute_import
from __future__ import division

import paddle

__all__ = [
    'InvertibleTransform',
]


class InvertibleTransform(paddle.nn.Layer):

    def _forward(self, *inputs, **kwargs):
        raise NotImplementedError()

    def _inverse(self, *inputs, **kwargs):
        raise NotImplementedError()

    def forward(self, *inputs, reverse=False, **kwargs):
        if not reverse:
            return self._forward(*inputs, **kwargs)
        else:
            return self._inverse(*inputs, **kwargs)
