#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import paddle
import paddle.fluid as fluid
import numpy as np
from scipy import stats
from scipy.special import logsumexp

import unittest

from tests.distributions import utils
from zhusuan.distributions.uniform import *


device = paddle.set_device('gpu')
paddle.disable_static(device)


class TestUniform(unittest.TestCase):

    def setUp(self):
        self._Uniform = lambda minval, maxval, **kwargs: Uniform(
            minval=minval, maxval=maxval,  **kwargs)

    def test_init_check_shape(self):
        try:
            Uniform(minval=paddle.zeros([2, 1]), maxval=paddle.ones([2, 4, 3]))
        except:
            raise ValueError("should be broadcastable to match")

        self._Uniform(paddle.zeros([32, 1], dtype='float32'),
                      paddle.ones([32, 1, 3], dtype='float32'))

    ## TODO: Define the value shape in Uniform module
    # def test_value_shape(self):
    #
    #     # dynamic
    #     unif = self._Uniform(paddle.zeros([], dtype='float32'),
    #                          paddle.ones([], dtype='float32'))
    #     paddle_int32 = paddle.cast(paddle.to_tensor([1]), dtype='int32').dtype
    #     self.assertTrue(unif._value_shape().dtype is paddle_int32)
    #     self.assertEqual(unif._value_shape(), [])
    #     self.assertEqual(unif._value_shape().dtype, paddle_int32)


    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, self._Uniform, np.zeros, np.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(
            self, self._Uniform, np.zeros, np.ones)

    def test_sample_reparameterized(self):
        minval = paddle.ones([2, 3])
        maxval = paddle.ones([2, 3])
        minval.stop_gradient = False
        maxval.stop_gradient = False
        unif_rep = self._Uniform(minval, maxval)
        samples = unif_rep.sample()
        minval_grads, maxval_grads = paddle.grad(outputs=[samples], inputs=[minval, maxval],
                                               allow_unused=True)
        self.assertTrue(minval_grads is not None)
        self.assertTrue(maxval_grads is not None)

        unif_no_rep = self._Uniform(minval, maxval, is_reparameterized=False)
        samples = unif_no_rep.sample()
        minval_grads, maxval_grads = paddle.grad(outputs=[samples],
                                               inputs=[minval, maxval],
                                               allow_unused=True)
        self.assertEqual(minval_grads, None)
        self.assertEqual(maxval_grads, None)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(
            self, self._Uniform, np.zeros, np.ones, np.zeros)

    def test_value(self):
        # with self.session(use_gpu=True):
        def _test_value(minval, maxval, given):
            minval = np.array(minval, np.float32)
            maxval = np.array(maxval, np.float32)
            given = np.array(given, np.float32)
            target_log_p = stats.uniform.logpdf(given, minval,
                                                maxval - minval)
            target_p = stats.uniform.pdf(given, minval, maxval - minval)

            minval = paddle.to_tensor(minval)
            maxval = paddle.to_tensor(maxval)
            given = paddle.to_tensor(given)
            unif = self._Uniform(minval, maxval)
            log_p = unif.log_prob(given)
            np.testing.assert_allclose(np.around(log_p.numpy(), decimals=6),
                                       np.around(target_log_p, decimals=6), rtol= 1e-03 )

            # # TODO: May add prob function to Uniform module in the future
            # p = unif.prob(given)
            # np.testing.assert_allclose(np.around(p.numpy(), decimals=6),
            #                            np.around(target_p, decimals=6))

        # Uniform semantics different from scipy at maxval.
        # TODO: Edit Uniform distribution module to support integer inputs
        # self.assertEqual(self._Uniform(0., 1.).log_prob(1).numpy(), -np.inf)
        _test_value(0., 1., [-1., 0., 0.5, 2.])
        _test_value([0.], [1.], [-1., 0., 0.5, 2.])
        _test_value([-1e10, -1], [1, 1e10], [0.])
        _test_value([0., -1.], [[[1., 2.], [3., 5.], [4., 9.]]], [7.])

    # TODO: Edit Uniform distribution module to support integer inputs
    # def test_check_numerics(self):
    #     unif = self._Uniform(0., [0., 1.], check_numerics=True)
    #     try:
    #         unif.log_prob(0.).numpy()
    #     except:
    #         raise ValueError("p.*Tensor had Inf")

    def test_dtype(self):
        utils.test_dtype_2parameter(self, self._Uniform)

    def test_distribution_shape(self):
        param1 = paddle.zeros([1])
        param2 = paddle.ones([1])
        distribution = self._Uniform(param1, param2)
        utils.test_and_save_distribution_img(distribution)

