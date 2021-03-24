# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy import stats
import numpy as np

import paddle

from zhusuan.variational.importance_weighted_objective import *
from zhusuan.distributions import Normal
from zhusuan.framework import BayesianNet

from tests.variational.utils import _kl_normal_normal

import unittest


# log_prob
class TestNode_Gen:
    def __init__(self, x_mean, x_std):
        super().__init__()
        self.x_mean = x_mean
        self.x_std = x_std
        self.observed = {}

    def observe(self, observed):
        self.observed = {}
        for k, v in observed.items():
            self.observed[k] = v

    def log_prob(self):
        norm = Normal(mean=self.x_mean, std=self.x_std)
        return norm.log_prob(self.observed['x'])


class TestNet_Gen(BayesianNet):
    def __init__(self, x_mean, x_std):
        super().__init__()
        self._nodes["test"] = TestNode_Gen(x_mean, x_std)

    def forward(self, observed):
        self._nodes["test"].observe(observed)
        return self


# latent
class TestNode_Var:
    def __init__(self, qx_samples, log_qx):
        self.tensor = qx_samples
        self.log_qx = log_qx

    def log_prob(self):
        return self.log_qx


class TestNet_Var(BayesianNet):
    def __init__(self, qx_samples, log_qx):
        super().__init__()
        self._nodes['x'] = TestNode_Var(qx_samples, log_qx)

    def forward(self, observed):
        return self


class TestImportanceWeightedObjective(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.RandomState(1)
        self._n1_samples = self._rng.standard_normal(size=(1, 1000)). \
            astype(np.float32)
        self._n3_samples = self._rng.standard_normal(1000).astype(np.float32)
        super(TestImportanceWeightedObjective, self).setUp()
        device = paddle.set_device('gpu')
        paddle.disable_static(device)

    def test_objective(self):
        log_qx_n1 = stats.norm.logpdf(self._n1_samples).astype(np.float32)
        qx_samples_n1 = paddle.to_tensor(self._n1_samples)
        log_qx_n1 = paddle.to_tensor(log_qx_n1)

        log_qx_n3 = stats.norm.logpdf(self._n3_samples).astype(np.float32)
        qx_samples_n3 = paddle.to_tensor(self._n3_samples)
        log_qx_n3 = paddle.to_tensor(log_qx_n3)

        def _check_kl_elbo(x_mean, x_std):
            _x_mean = paddle.to_tensor(x_mean, stop_gradient=True)
            _x_std = paddle.to_tensor(x_std, stop_gradient=True)
            model = ImportanceWeightedObjective(TestNet_Gen(_x_mean, _x_std), TestNet_Var(qx_samples_n1, log_qx_n1),
                                                axis=0)
            lower_bound = -model({}).numpy()
            analytic_lower_bound = -_kl_normal_normal(paddle.to_tensor(0.), paddle.to_tensor(1.), _x_mean, _x_std) \
                .numpy()
            # print(lower_bound)
            # print(analytic_lower_bound)
            self.assertAlmostEqual(lower_bound[0], analytic_lower_bound[0], delta=1e-2)

        _check_kl_elbo(0., 1.)
        _check_kl_elbo(2., 3.)

        def _check_monotonous_elbo(x_mean, x_std):
            _x_mean = paddle.to_tensor(x_mean, stop_gradient=True)
            _x_std = paddle.to_tensor(x_std, stop_gradient=True)
            model = ImportanceWeightedObjective(TestNet_Gen(_x_mean, _x_std), TestNet_Var(qx_samples_n3, log_qx_n3),
                                                axis=0)
            lower_bound = -model({}).numpy()
            analytic_lower_bound = -_kl_normal_normal(paddle.to_tensor(0.), paddle.to_tensor(1.), _x_mean, _x_std) \
                .numpy()
            # print(lower_bound)
            # print(analytic_lower_bound)
            self.assertTrue(lower_bound[0] > analytic_lower_bound[0] - 1e-6)

        _check_monotonous_elbo(0., 1.)
        _check_monotonous_elbo(2., 3.)

    def test_sgvb(self):
        eps_samples = paddle.to_tensor(self._n1_samples, dtype='float32')
        mu = paddle.to_tensor(2., stop_gradient=False)
        sigma = paddle.to_tensor(3., stop_gradient=False)
        qx_samples = eps_samples * sigma + mu
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        def _check_sgvb(x_mean, x_std, threshold):
            _x_mean = paddle.to_tensor(x_mean, stop_gradient=True)
            _x_std = paddle.to_tensor(x_std, stop_gradient=True)
            model = ImportanceWeightedObjective(TestNet_Gen(_x_mean, _x_std), TestNet_Var(qx_samples, log_qx), axis=0)
            sgvb_cost = model({})
            sgvb_grads = paddle.grad(sgvb_cost, [mu, sigma], retain_graph=True)
            sgvb_grads = [paddle.squeeze(sgvb_grads[0], axis=[0])] + [sgvb_grads[1]]
            sgvb_grads = paddle.concat(sgvb_grads).numpy()
            true_cost = _kl_normal_normal(mu, sigma, x_mean, x_std)
            true_grads = paddle.grad(true_cost, [mu, sigma], retain_graph=True)
            true_grads = paddle.concat(true_grads).numpy()
            print('sgvb_grads: ', sgvb_grads)
            print('true_grads: ', true_grads)
            np.testing.assert_allclose(sgvb_grads, true_grads, threshold, threshold)

        _check_sgvb(0., 1., 0.04)
        _check_sgvb(2., 3., 0.02)

    def test_vimco(self):
        eps_samples = paddle.to_tensor(self._n3_samples, dtype='float32')
        mu = paddle.to_tensor(2., stop_gradient=False)
        sigma = paddle.to_tensor(3., stop_gradient=False)
        qx_samples = eps_samples * sigma + mu
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        v_qx_samples = eps_samples * sigma.detach() + mu.detach()
        v_log_qx = norm.log_prob(v_qx_samples)
        mu.stop_gradient = False
        sigma.stop_gradient = False

        def _check_vimco(x_mean, x_std, threshold):
            _x_mean = paddle.to_tensor(x_mean, stop_gradient=True)
            _x_std = paddle.to_tensor(x_std, stop_gradient=True)
            model_sgvb = ImportanceWeightedObjective(TestNet_Gen(_x_mean, _x_std), TestNet_Var(qx_samples, log_qx),
                                                     axis=0, estimator='sgvb')
            model_vimco = ImportanceWeightedObjective(TestNet_Gen(_x_mean, _x_std), TestNet_Var(v_qx_samples, v_log_qx),
                                                      axis=0, estimator='vimco')
            vimco_cost = model_vimco({})
            vimco_grads = paddle.grad(vimco_cost, [mu, sigma], retain_graph=True)
            vimco_grads = paddle.concat(vimco_grads).numpy()
            sgvb_cost = model_sgvb({})
            sgvb_grads = paddle.grad(sgvb_cost, [mu, sigma], retain_graph=True)
            sgvb_grads = paddle.concat(sgvb_grads).numpy()
            print("vimco_grads: ", vimco_grads)
            print("sgvb_grads: ", sgvb_grads)
            np.testing.assert_allclose(vimco_grads, sgvb_grads, threshold, threshold)
        _check_vimco(0., 1., 1e-2)
        _check_vimco(2., 3., 1e-6)


if __name__ == '__main__':
    unittest.main()