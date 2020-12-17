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
from zhusuan.distributions.beta import *


device = paddle.set_device('gpu')
paddle.disable_static(device)


class TestBeta(unittest.TestCase):

    def setUp(self):
        self._Beta = lambda alpha, beta, **kwargs: Beta(
            alpha=alpha, beta=beta,  **kwargs)

    def test_init_check_shape(self):
        try:
            Beta(alpha=paddle.ones([2, 1]), beta=paddle.ones([2, 4, 3]))
        except:
            ValueError("should be broadcastable to match")

        self._Beta(paddle.zeros([32, 1], dtype='float32'),
                    paddle.ones([32, 1, 3], dtype='float32'))

    # # TODO: Define the value shape in Beta module
    # def test_value_shape(self):
    #     # dynamic
    #     dist = self._Beta(paddle.zeros([], dtype='float32'),
    #                        paddle.ones([], dtype='float32'))
    #     paddle_int32 = paddle.cast(paddle.to_tensor([1]), dtype='int32').dtype
    #     self.assertTrue(dist._value_shape().dtype is paddle_int32)
    #     self.assertEqual(dist._value_shape(), [])
    #     self.assertEqual(dist._value_shape().dtype, paddle_int32)

    def test_batch_shape(self):
        utils.test_batch_shape_2parameter_univariate(
            self, self._Beta, np.ones, np.ones)

    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(
            self, self._Beta, np.ones, np.ones)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(
            self, self._Beta, np.ones, np.ones, np.ones)

    def test_value(self):
        def _test_value(alpha, beta, given):
            alpha = np.array(alpha, np.float32)
            beta = np.array(beta, np.float32)
            given = np.array(given, np.float32)
            target_log_p = stats.beta.logpdf(given, alpha, beta)
            target_p = stats.beta.pdf(given, alpha, beta)

            alpha = paddle.to_tensor(alpha)
            beta = paddle.to_tensor(beta)
            given = paddle.to_tensor(given)

            dist = self._Beta(alpha, beta)
            log_p = dist.log_prob(given)
            np.testing.assert_allclose(np.around(log_p.numpy(), decimals=6),
                                       np.around(target_log_p, decimals=6), rtol= 1e-03 )

            # # TODO: May add prob function to Beta module in the future
            # p = dist.prob(given)
            # np.testing.assert_allclose(np.around(p.numpy(), decimals=6),
            #                            np.around(target_p, decimals=6), rtol=1e-03)

        # _test_value(1., 1., [0., 0.5, 1.])
        _test_value([0.5, 5., 1., 2., 2.],
                    [0.5, 1., 3., 2., 5.],
                    np.transpose([np.arange(0.1, 1, 0.1)]))
        _test_value([[1e-8], [1e8]], [[1., 1e8], [1e-8, 1.]], [0.7])

    def test_check_numerics(self):
        try:
            alpha = paddle.to_tensor(1.)
            beta = paddle.to_tensor(1.)
            given = paddle.to_tensor(0.)
            dist = self._Beta(alpha, beta, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("log\(given\).*Tensor had Inf")
        try:
            alpha = paddle.to_tensor(1.)
            beta = paddle.to_tensor(1.)
            given = paddle.to_tensor(1.)
            dist = self._Beta(alpha, beta, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("log\(1 - given\).*Tensor had Inf")
        try:
            alpha = paddle.to_tensor(2.)
            beta = paddle.to_tensor(-1.)
            given = paddle.to_tensor(.5)
            dist = self._Beta(alpha, beta, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("lgamma\(beta\).*Tensor had Inf")
        try:
            alpha = paddle.to_tensor(0.)
            beta = paddle.to_tensor(1.)
            given = paddle.to_tensor(.5)
            dist = self._Beta(alpha, beta, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("lgamma\(alpha\).*Tensor had Inf")

    def test_dtype(self):
        utils.test_dtype_2parameter(self, self._Beta)

    def test_distribution_shape(self):
        # you can try like: param2 = paddle.ones([1])*4.
        #  or param2 = paddle.ones([1])*3. to see the difference of the pdf
        param1 = paddle.ones([1])*2.
        param2 = paddle.ones([1])*5.
        distribution = self._Beta(param1, param2)
        utils.test_and_save_distribution_img(distribution)

