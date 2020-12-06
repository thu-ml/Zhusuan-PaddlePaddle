#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import paddle
import paddle.fluid as fluid
import numpy as np
import math


__all__ = [
    'open_interval_standard_uniform',
    'log_combination'
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


def log_combination(n, ks):
    """
    Compute the log combination function.
    .. math::
        \\log \\binom{n}{k_1, k_2, \\dots} = \\log n! - \\sum_{i}\\log k_i!
    :param n: A N-D `float` Tensor. Can broadcast to match `tf.shape(ks)[:-1]`.
    :param ks: A (N + 1)-D `float` Tensor. Each slice `[i, j, ..., k, :]` is
        a vector of `[k_1, k_2, ...]`.
    :return: A N-D Tensor of type same as `n`.
    """
    # TODO: Paddle do not have lgamma module. Here we use Math package
    lgamma_n_plus_1 = paddle.to_tensor(list(map(lambda x: math.lgamma(x), n + 1)))
    lgamma_ks_plus_1 = paddle.to_tensor(list(map(lambda x: math.lgamma(x), ks + 1)))
    return lgamma_n_plus_1 - fluid.layers.reduce_sum(lgamma_ks_plus_1, dim=-1)


