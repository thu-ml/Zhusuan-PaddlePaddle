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
from zhusuan.distributions.poisson import *


device = paddle.set_device('gpu')
paddle.disable_static(device)


class TestPoisson(unittest.TestCase):

    def setUp(self):
        self._Poisson = lambda rate,  **kwargs: Poisson(
            rate=rate,  **kwargs)

    # # TODO: Define the value shape in Beta module
    # def test_value_shape(self):
    #     # dynamic
    #     dist = self._Poisson(paddle.ones([], dtype='float32'))
    #     paddle_int32 = paddle.cast(paddle.to_tensor([1]), dtype='int32').dtype
    #     self.assertTrue(dist._value_shape().dtype is paddle_int32)
    #     self.assertEqual(dist._value_shape(), [])
    #     self.assertEqual(dist._value_shape().dtype, paddle_int32)

    def test_batch_shape(self):
        utils.test_batch_shape_1parameter(
            self, self._Poisson, np.ones, is_univariate=True)

    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_same(
            self, self._Poisson, np.ones)

    def test_log_prob_shape(self):
        utils.test_1parameter_log_prob_shape_same(
            self, self._Poisson, np.ones, np.ones)

    def test_value(self):

        def _test_value(rate, given):
            rate = np.array(rate, np.float32)
            given = np.array(given, np.float32)
            target_log_p = stats.poisson.logpmf(given, rate)
            target_p = stats.poisson.pmf(given, rate)

            rate = paddle.to_tensor(rate)
            given = paddle.to_tensor(given)

            poisson = self._Poisson(rate)
            log_p = poisson.log_prob(given)

            target_log_p = target_log_p.astype(log_p.numpy().dtype)
            np.testing.assert_allclose(np.around(log_p.numpy(), decimals=6),
                                       np.around(target_log_p, decimals=6), rtol= 1e-03 )

            # # TODO: May add prob function to Poisson module in the future
            # p = poisson.prob(given)
            # np.testing.assert_allclose(np.around(p.numpy(), decimals=6),
            #                            np.around(target_p, decimals=6), rtol= 1e-03 )

        _test_value(1, [0, 1, 2, 3, 4, 5, 6])
        _test_value([5, 1, 5], [0, 0, 1])
        _test_value([10000, 1], [[100, 0], [0, 100]])
        _test_value([[1, 10, 100], [999, 99, 9]],
                    np.ones([3, 1, 2, 3], dtype=np.int32))
        _test_value([[1, 10, 100], [999, 99, 9]],
                    100 * np.ones([3, 1, 2, 3], dtype=np.int32))

    def test_check_numerics(self):
        try:
            rate = paddle.to_tensor(1.)
            given = paddle.to_tensor(-2)
            dist = self._Poisson(rate, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("lgamma\(given \+ 1\).*Tensor had Inf")
        try:
            rate = paddle.to_tensor(-1.)
            given = paddle.to_tensor(1)
            dist = self._Poisson(rate, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("log\(rate\).*Tensor had NaN")

    def test_dtype(self):
        utils.test_dtype_1parameter_discrete(self, self._Poisson)

    def test_distribution_shape(self):
        # you can try like: param = paddle.ones([1])*10
        #  or param = paddle.ones([1])*20 to see the difference of the pdf
        param = paddle.ones([1])*15
        distribution = self._Poisson(param)
        utils.test_and_save_distribution_img(distribution)

