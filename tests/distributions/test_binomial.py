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
from zhusuan.distributions.binomial import *


device = paddle.set_device('gpu')
paddle.disable_static(device)


class TestBinomial(unittest.TestCase):

    def setUp(self):
        self._Binomial = lambda probs, n_experiments,  **kwargs: Binomial(
            probs=probs, n_experiments=n_experiments,  **kwargs)

    def test_init_n(self):
        dist = self._Binomial(paddle.ones([2]), 10)
        self.assertTrue(isinstance(dist.n_experiments, int))
        self.assertEqual(dist.n_experiments, 10)
        try:
            _ = self._Binomial(paddle.ones([2]), 0)
        except:
            raise ValueError("must be positive")
        try:
            logits = paddle.cast(paddle.to_tensor([1.]), 'float32')
            n_experiments = paddle.cast(paddle.to_tensor([10]), 'int32')
            dist2 = self._Binomial(logits, n_experiments)
            dist2.n_experiments.numpy()
        except:
            raise ValueError("should be a scalar")
        try:
            logits = paddle.cast(paddle.to_tensor([1.]), 'float32')
            n_experiments = paddle.cast(paddle.to_tensor([0]), 'int32')
            dist2 = self._Binomial(logits, n_experiments)
            dist2.n_experiments.numpy()
        except:
            raise ValueError("must be positive")

    # # TODO: Define the value shape in Binomial module
    # def test_value_shape(self):
    #     # dynamic
    #     dist = self._Binomial(paddle.ones([], dtype='float32'), 0)
    #     paddle_int32 = paddle.cast(paddle.to_tensor([1]), dtype='int32').dtype
    #     self.assertTrue(dist._value_shape().dtype is paddle_int32)
    #     self.assertEqual(dist._value_shape(), [])
    #     self.assertEqual(dist._value_shape().dtype, paddle_int32)

    def test_batch_shape(self):
        def _distribution(param):
            return self._Binomial(param, 10)
        utils.test_batch_shape_1parameter(
            self, _distribution, np.ones, is_univariate=True)

    def test_sample_shape(self):
        def _distribution(param):
            return self._Binomial(param, 10)
        utils.test_1parameter_sample_shape_same(
            self, _distribution, np.ones)

    def test_log_prob_shape(self):
        def _distribution(param):
            return self._Binomial(param, 10)
        utils.test_1parameter_log_prob_shape_same(
            self, _distribution, np.ones, np.ones)

    def test_value(self):

        def _test_value(logits, n_experiments, given):
            logits = np.array(logits, np.float64)
            given = np.array(given, np.float64)
            target_log_p = stats.binom.logpmf(
                given, n_experiments, 1 / (1. + np.exp(-logits)))
            target_p = stats.binom.pmf(
                given, n_experiments, 1 / (1. + np.exp(-logits)))

            logits = paddle.to_tensor(logits)
            given = paddle.to_tensor(given)

            binomial = self._Binomial(logits, n_experiments)
            log_p = binomial.log_prob(given)
            target_log_p = target_log_p.astype(log_p.numpy().dtype)
            np.testing.assert_allclose(np.around(log_p.numpy(), decimals=6),
                                       np.around(target_log_p, decimals=6), rtol= 1e-02 )
            # When transforming log-odds to probabilities, there will be
            # some loss of precision. Besides, the log_prob may become
            # very large. So the absolute tolerance (atol) can't be the
            # default value (1e-06).

            # # TODO: May add prob function to Binomial module in the future
            # p = binomial.prob(given)
            # np.testing.assert_allclose(np.around(p.numpy(), decimals=6),
            #                            np.around(target_p, decimals=6), rtol= 1e-02 )

        _test_value(0., 6, [0, 1, 2, 3, 4, 5, 6])
        _test_value([5., -1., 5.], 2, [0, 0, 1])
        _test_value([10., -10., 0.], 200,
                    [[10, 10, 10], [190, 190, 190]])
        _test_value([[1., 5., 10.], [-1., -5., -10.]], 20,
                    np.ones([3, 1, 2, 3], dtype=np.int32))
        _test_value([[1., 5., 10.], [-1., -5., -10.]], 20,
                    19 * np.ones([3, 1, 2, 3], dtype=np.int32))

    def test_check_numerics(self):
        try:
            logits = paddle.to_tensor(1.)
            given = paddle.to_tensor(-2)
            dist = self._Binomial(logits, 10, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("lgamma\(given \+ 1\).*Tensor had Inf")
        try:
            logits = paddle.to_tensor(1.)
            given = paddle.to_tensor(12)
            dist = self._Binomial(logits, 10, check_numerics=True)
            dist.log_prob(given).numpy()
        except:
            raise ValueError("lgamma\(n - given \+ 1\).*Tensor had Inf")

    def test_dtype(self):
        def _distribution(param, **kwargs):
            return self._Binomial(param, 10, **kwargs)
        utils.test_dtype_1parameter_discrete(self, _distribution)

    def test_distribution_shape(self):
        # you can try like: param = paddle.ones([1])*.5
        #  or n = 40 to see the difference of the pdf
        param = paddle.ones([1])*.7
        n = 20
        distribution = self._Binomial(param, n)
        utils.test_and_save_distribution_img(distribution)

