#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import paddle
import numpy as np


__all__ = [
    'open_interval_standard_uniform'
]

def open_interval_standard_uniform(shape, dtype):
    """
    Return samples from uniform distribution in unit open interval (0, 1).
    :param shape: The shape of generated samples.
    :param dtype: The dtype of generated samples.
    :return: A Tensor of samples.
    """
    return paddle.cast(paddle.uniform(shape=shape,
                                      min=np.finfo(dtype.as_numpy_dtype).tiny,
                                      max=1.),
                       dtype=dtype)
