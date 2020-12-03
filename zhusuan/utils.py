#!/usr/bin/env python
# -*- coding: utf-8 -*-
import paddle
import paddle.fluid as fluid

def log_mean_exp(x, axis=None, keepdims=False):
    """
    Tensorflow numerically stable log mean of exps across the `axis`.
    :param x: A Tensor.
    :param axis: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keepdims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.
    :return: A Tensor after the computation of log mean exp along given axes of
        x.
    """
    x_max = fluid.layers.reduce_max(x, axis=axis, keepdims=True)
    ret = paddle.log(fluid.layers.reduce_mean(paddle.exp(x - x_max), axis=axis,
                                keepdims=True)) + x_max
    if not keepdims:
        ret = fluid.layers.reduce_mean(ret, axis=axis)
    return ret



