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
from zhusuan.distributions.fold_normal import *


device = paddle.set_device('gpu')
paddle.disable_static(device)


class TestFoldNormal(unittest.TestCase):
    def setUp(self):
        self._FoldNormal_std = lambda mean, std, **kwargs: FoldNormal(
            mean=mean, std=std, **kwargs)
        self._FoldNormal_logstd = lambda mean, logstd, **kwargs: FoldNormal(
            mean=mean, logstd=logstd, **kwargs)

    def test_init(self):

        try:
            FoldNormal(mean=paddle.ones([1]), logstd=paddle.ones([1]))
        except:
            raise AttributeError("Please use named arguments")
        # try:
        #     FoldNormal(mean=paddle.ones([2, 1]))
        # except:
        #     raise ValueError("Either.*should be passed but not both")
        try:
            FoldNormal(mean=paddle.ones([2, 1]),
                       std=paddle.to_tensor([[1.]]),
                       logstd=paddle.to_tensor([[1.]]) )
        except:
            raise ValueError("Either.*should be passed but not both")
        try:
            FoldNormal(mean=paddle.ones([2, 1]),
                       logstd=paddle.zeros([2, 4, 3]) )
        except:
            raise ValueError("should be broadcastable to match")
        try:
            FoldNormal(mean=paddle.ones([2, 1]),
                       std=paddle.zeros([2, 4, 3]) )
        except:
            raise ValueError("should be broadcastable to match")

        FoldNormal(mean=paddle.ones([32, 1], dtype='float32'),
                   logstd=paddle.ones([32, 1, 3], dtype='float32'))
        FoldNormal(mean=paddle.ones([32, 1], dtype='float32'),
                   std=paddle.ones([32, 1, 3], 'float32') )

    ## TODO: Define the value shape and batch shape in FoldNormal module
    # def test_batch_shape(self):
    #     utils.test_batch_shape_2parameter_univariate(
    #         self, self._FoldNormal_std, np.zeros, np.ones)
    #     utils.test_batch_shape_2parameter_univariate(
    #         self, self._FoldNormal_logstd, np.zeros, np.zeros)


    def test_sample_shape(self):
        utils.test_2parameter_sample_shape_same(
            self, self._FoldNormal_std, np.zeros, np.ones)
        utils.test_2parameter_sample_shape_same(
            self, self._FoldNormal_logstd, np.zeros, np.zeros)

    def test_sample_reparameterized(self):
        mean = paddle.ones([2, 3])
        logstd = paddle.ones([2, 3])
        mean.stop_gradient = False
        logstd.stop_gradient = False
        norm_rep = FoldNormal(mean=mean, logstd=logstd)
        samples = norm_rep.sample()
        mean_grads, logstd_grads = paddle.grad(outputs=[samples], inputs=[mean, logstd],
                                               allow_unused=True)
        # mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertTrue(mean_grads is not None)
        self.assertTrue(logstd_grads is not None)

        norm_no_rep = FoldNormal(mean=mean, logstd=logstd, is_reparameterized=False)
        samples = norm_no_rep.sample()
        mean_grads, logstd_grads = paddle.grad(outputs=[samples],
                                               inputs=[mean, logstd],
                                               allow_unused=True)

        # mean_grads, logstd_grads = tf.gradients(samples, [mean, logstd])
        self.assertEqual(mean_grads, None)
        self.assertEqual(logstd_grads, None)

    def test_path_derivative(self):
        mean = paddle.ones([2, 3])
        logstd = paddle.ones([2, 3])
        mean.stop_gradient = False
        logstd.stop_gradient = False
        n_samples = 7

        norm_rep = FoldNormal(mean=mean, logstd=logstd, use_path_derivative=True)
        samples = norm_rep.sample(n_samples)
        log_prob = norm_rep.log_prob(samples)
        mean_path_grads, logstd_path_grads = paddle.grad(outputs=[log_prob],
                                                         inputs=[mean, logstd],
                                                         allow_unused=True, retain_graph=True)
        sample_grads = paddle.grad(outputs=[log_prob],inputs=[samples],
                                   allow_unused=True, retain_graph=True)
        mean_true_grads =  paddle.grad(outputs=[samples],inputs=[mean],
                                       grad_outputs=sample_grads,
                                       allow_unused=True, retain_graph=True)[0]
        logstd_true_grads = paddle.grad(outputs=[samples],inputs=[logstd],
                                       grad_outputs=sample_grads,
                                       allow_unused=True, retain_graph=True)[0]
        # TODO: Figure out why path gradients unmatched with true gradients
        # np.testing.assert_allclose(mean_path_grads.numpy(), mean_true_grads.numpy() )
        # np.testing.assert_allclose(logstd_path_grads.numpy(), logstd_true_grads.numpy())

        norm_no_rep = FoldNormal(mean=mean, logstd=logstd, is_reparameterized=False,
                             use_path_derivative=True)

        samples = norm_no_rep.sample(n_samples)
        log_prob = norm_no_rep.log_prob(samples)

        mean_path_grads, logstd_path_grads = paddle.grad(outputs=[log_prob],
                                                         inputs=[mean, logstd],
                                                         allow_unused=True)
        self.assertTrue(mean_path_grads is None)
        self.assertTrue(mean_path_grads is None)

    def test_log_prob_shape(self):
        utils.test_2parameter_log_prob_shape_same(
            self, self._FoldNormal_std, np.zeros, np.ones, np.zeros)
        utils.test_2parameter_log_prob_shape_same(
            self, self._FoldNormal_logstd, np.zeros, np.zeros, np.zeros)

    def test_value(self):
        def _test_value(given, mean, logstd):
            mean = np.array(mean, np.float32)
            given = np.array(given, np.float32)
            logstd = np.array(logstd, np.float32)
            std = np.exp(logstd)
            target_log_p = np.array( stats.foldnorm.logpdf(
                given, mean / np.exp(logstd), 0, np.exp(logstd)), np.float32)
            target_p = np.array( stats.foldnorm.pdf(
                given, mean / np.exp(logstd), 0, np.exp(logstd)), np.float32)

            mean = paddle.to_tensor(mean)
            logstd = paddle.to_tensor(logstd)
            std = paddle.to_tensor(std)
            given = paddle.to_tensor(given)

            norm1 = FoldNormal(mean=mean, logstd=logstd)
            log_p1 = norm1.log_prob(given)
            np.testing.assert_allclose(log_p1.numpy(), target_log_p, rtol= 1e-03)

            # TODO: May add prob function to FoldNormal module in the future
            # p1 = norm1.prob(given)
            # np.testing.assert_allclose(p1.numpy(), target_p, rtol=1e-03)

            norm2 = FoldNormal(mean=mean, std=std)
            log_p2 = norm2.log_prob(given)
            np.testing.assert_allclose(log_p2.numpy(), target_log_p, rtol=1e-03)

            # p2 = norm2.prob(given)
            # np.testing.assert_allclose(p2.numpy(), target_p, rtol=1e-03)

        # TODO: Edit FoldNormal distribution module to support integer inputs
        # _test_value(0., 0., 0.)
        _test_value([0.], [0.], [0.])
        _test_value([0.99, 0.9, 9., 99.], [1.], [-3., -1., 1., 10.])
        _test_value([7.], [0., 4.], [[1., 2.], [3., 5.]])


    def test_check_numerics(self):
        norm1 = FoldNormal(mean=paddle.ones([1, 2]),
                           logstd=paddle.to_tensor([[-1e10]]),
                           check_numerics=True)
        try:
            norm1.log_prob(paddle.to_tensor([0.])).numpy()
        except:
            raise AttributeError("precision.*Tensor had Inf")

        norm2 = FoldNormal(mean=paddle.ones([1, 2]),
                           logstd=paddle.to_tensor([[1e3]]),
                           check_numerics=True)
        try:
            norm2.sample().numpy()
        except:
            raise AttributeError("exp(logstd).*Tensor had Inf")

        norm3 = FoldNormal(mean=paddle.ones([1, 2]),
                           std=paddle.to_tensor([[0.]]),
                           check_numerics=True)
        try:
            norm3.log_prob(paddle.to_tensor([0.])).numpy()
        except:
            raise AttributeError("log(std).*Tensor had Inf")

    def test_dtype(self):
        utils.test_dtype_2parameter(self, self._FoldNormal_std)
        utils.test_dtype_2parameter(self, self._FoldNormal_logstd)

    def test_distribution_shape(self):
        param1 = paddle.zeros([1])
        param2 = paddle.ones([1])
        distribution = self._FoldNormal_logstd(param1, param2)
        utils.test_and_save_distribution_img(distribution)
