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
from zhusuan.distributions.bernoulli import *


device = paddle.set_device('gpu')
paddle.disable_static(device)


class TestBernoulli(unittest.TestCase):

    def setUp(self):
        self._Bernoulli = lambda probs, **kwargs: Bernoulli(probs=probs,  **kwargs)


    ## TODO: Define the value shape and batch shape in Bernoulli module
    # def test_value_shape(self):
    #
    #     # get value shape
    #     norm = Bernoulli(paddle.cast(paddle.to_tensor([]), 'float32'))
    #     self.assertEqual(norm.get_value_shape(), [])
    #
    #     # dynamic
    #     self.assertTrue(norm._value_shape().dtype is 'int32')
    #     self.assertEqual(norm._value_shape(), [])
    #
    #     self.assertEqual(norm._value_shape().dtype, 'int32')

    ## TODO: Define the value shape and batch shape in Bernoulli module
    # def test_batch_shape(self):
    #     utils.test_batch_shape_1parameter(
    #         self, Bernoulli, np.zeros, is_univariate=True)


    def test_sample_shape(self):
        utils.test_1parameter_sample_shape_same(
            self, self._Bernoulli, np.zeros)

    def test_log_prob_shape(self):
        utils.test_1parameter_log_prob_shape_same(
            self, self._Bernoulli, np.zeros, np.zeros)

    def test_value(self):
        def _test_value(logits, given):
            logits = np.array(logits, np.float32)
            given = np.array(given, np.float32)

            target_log_p = stats.bernoulli.logpmf(
                given, 1. / (1. + np.exp(-logits)))
            target_p = stats.bernoulli.pmf(
                given, 1. / (1. + np.exp(-logits)))

            logits = paddle.to_tensor(logits)
            given = paddle.to_tensor(given)

            bernoulli = self._Bernoulli(logits)
            log_p = bernoulli.log_prob(given)
            np.testing.assert_allclose(np.around(log_p.numpy(), decimals=6),
                                       np.around(target_log_p, decimals=6), rtol=1e-03)

            # TODO: May add prob function to Bernoulli module in the future
            # p = bernoulli.prob(given)
            # np.testing.assert_allclose(p.numpy(), target_p, rtol=1e-03)

        # TODO: Edit Bernoulli distribution module to support integer inputs
        # _test_value(0., [0, 1])
        _test_value([0.], [0, 1])
        _test_value([-50., -10., -50.], [1, 1, 0])
        _test_value([0., 4.], [[0, 1], [0, 1]])
        _test_value([[2., 3., 1.], [5., 7., 4.]],
                    np.ones([3, 2, 3], dtype=np.int32))
        # TODO: Edit Bernoulli distribution module to slove the broadcast issue
        #  when len(given.shape) - len(logits.shape) >= 2
        # _test_value([[2., 3., 1.], [5., 7., 4.]],
        #             np.ones([3, 1, 2, 3], dtype=np.int32))

    def test_dtype(self):
        utils.test_dtype_1parameter_discrete(self, self._Bernoulli)

    def test_distribution_shape(self):
        param = paddle.ones([1])
        distribution = self._Bernoulli(param)
        utils.test_and_save_distribution_img(distribution)

