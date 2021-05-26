from __future__ import absolute_import
from __future__ import division

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
import numpy as np
import scipy

from zhusuan.transforms.invertible.base import InvertibleTransform

__all__ = [
    'Scaling',
]


class Scaling(InvertibleTransform):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = self.create_parameter(shape=[1, dim],
                                           default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.add_parameter("log_scale", self.scale)

    def _forward(self, x, **kwargs):
        log_det_J = self.log_scale.clone()
        x *= paddle.exp(self.log_scale)
        return x, log_det_J

    def _inverse(self, y, **kwargs):
        log_det_J = self.log_scale.clone()
        y *= paddle.exp(-self.log_scale)
        return y, log_det_J


