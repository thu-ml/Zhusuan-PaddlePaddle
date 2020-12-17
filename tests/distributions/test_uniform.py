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

    ## TODO: Define the value shape in Categorical module
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







    # def test_init_check_shape(self):
    #     try:
    #         Categorical(probs=paddle.zeros([1]))
    #     except:
    #         ValueError("should have rank")
    #
    # def test_init_n_categories(self):
    #     cat = self._Categorical(paddle.ones([10]))
    #     self.assertTrue(isinstance(cat.n_categories, int))
    #     self.assertEqual(cat.n_categories, 10)
    #     cat2 = self._Categorical( paddle.ones( [3, 1], dtype='float32') )
    #     self.assertTrue(cat2.n_categories is not None)
    #
    #     logits = paddle.ones([10], 'float32')
    #     cat3 = self._Categorical(logits)
    #     self.assertEqual( cat3.n_categories, 10)
    #     try:
    #         ## TODO: Edit Categorical distribution module to support integer inputs
    #         # logits = paddle.ones(1.)
    #         logits = paddle.ones([1.])
    #         cat3 = self._Categorical(logits)
    #         cat3.n_categories
    #     except:
    #         raise AttributeError("should have rank")
    #
    # ## TODO: Define the value shape in Categorical module
    # # def test_value_shape(self):
    # #
    # #     # get value shape
    # #     norm = self._Categorical(paddle.cast(paddle.to_tensor([]), 'float32'))
    # #     self.assertEqual(norm.get_value_shape(), [])
    # #
    # #     # dynamic
    # #     self.assertTrue(norm._value_shape().dtype is 'int32')
    # #     self.assertEqual(norm._value_shape(), [])
    # #
    # #     self.assertEqual(norm._value_shape().dtype, 'int32')
    #
    # def test_batch_shape(self):
    #     # dynamic
    #     def _test_dynamic(logits_shape):
    #         logits = paddle.zeros(logits_shape, dtype='float32')
    #         cat = self._Categorical(logits)
    #         self.assertEqual( cat.batch_shape, logits_shape[:-1])
    #
    #     _test_dynamic([2])
    #     _test_dynamic([2, 3])
    #     _test_dynamic([2, 1, 4])
    #
    # def test_sample_shape(self):
    #
    #     def _test_dynamic(logits_shape, n_samples, target_shape):
    #         logits = paddle.ones(logits_shape, dtype='float32')
    #         cat = self._Categorical(logits)
    #         samples = cat.sample(n_samples)
    #         self.assertEqual(samples.shape, target_shape)
    #
    #     _test_dynamic([2], 1, [1])
    #     _test_dynamic([2, 3], 1, [1, 2])
    #     _test_dynamic([1, 3], 2, [2, 1])
    #     _test_dynamic([2, 1, 5], 3, [3, 2, 1])
    #
    # def test_log_prob_shape(self):
    #
    #     def _test_dynamic(logits_shape, given_shape, target_shape):
    #         logits = paddle.zeros(logits_shape, dtype='float32')
    #         cat = self._Categorical(logits)
    #         given = paddle.zeros(given_shape, dtype='int32')
    #         log_p = cat.log_prob(given)
    #         pred_shape = log_p.shape if not log_p is None else None
    #         self.assertEqual(pred_shape, target_shape)
    #
    #     _test_dynamic([2, 3, 3], [1, 3], [2, 3])
    #     _test_dynamic([1, 3, 4], [2, 2, 3], [2, 2, 3])
    #     _test_dynamic([1, 5, 1], [1, 2, 3, 1], [1, 2, 3, 5])
    #     # try:
    #     #     _test_dynamic([2, 3, 5], [1, 2], None)
    #     # except:
    #     #     raise AttributeError("Incompatible shapes")
    #
    # def test_value(self):
    #     def _test_value(logits, given):
    #         logits = np.array(logits, np.float32)
    #         normalized_logits = logits - logsumexp(
    #             logits, axis=-1, keepdims=True)
    #         given = np.array(given, np.int32)
    #
    #         def _one_hot(x, depth):
    #             n_elements = x.size
    #             ret = np.zeros((n_elements, depth))
    #             ret[np.arange(n_elements), x.flat] = 1
    #             return ret.reshape(list(x.shape) + [depth])
    #
    #         target_log_p = np.sum(_one_hot(
    #             given, logits.shape[-1]) * normalized_logits, -1)
    #
    #         target_p = np.sum(_one_hot(
    #             given, logits.shape[-1]) * np.exp(normalized_logits), -1)
    #
    #         logits = paddle.to_tensor(logits)
    #         given = paddle.to_tensor(given)
    #         cat = self._Categorical(logits)
    #
    #         log_p = cat.log_prob(given)
    #         np.testing.assert_allclose(np.around(log_p.numpy(), decimals=6),
    #                                    np.around(target_log_p, decimals=6))
    #
    #         # TODO: May add prob function to Categorical module in the future
    #         # p = cat.prob(given)
    #         # np.testing.assert_allclose(np.around(p.numpy(), decimals=6),
    #         #                            np.around(target_p, decimals=6))
    #
    #     _test_value([0.], [0, 0, 0])
    #     _test_value([-50., -10., -50.], [0, 1, 2, 1])
    #     _test_value([0., 4.], [[0, 1], [0, 1]])
    #     _test_value([[2., 3., 1.], [5., 7., 4.]],
    #                 np.ones([3, 1, 1], dtype=np.int32))
    #
    # def test_dtype(self):
    #     utils.test_dtype_1parameter_discrete(
    #         self, self._Categorical, allow_16bit=False)
    #
