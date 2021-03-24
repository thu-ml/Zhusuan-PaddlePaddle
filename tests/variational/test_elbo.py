# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy import stats
import numpy as np
import paddle
import paddle.fluid as fluid

from zhusuan.variational.elbo import *
from zhusuan.distributions import Normal
from zhusuan.framework import BayesianNet

from tests.variational.utils import _kl_normal_normal

import numpy as np

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

class TestEvidenceLowerBound(unittest.TestCase):
    def setUp(self):
        self._rng = np.random.RandomState(1)
        self._n01_1e5 = self._rng.standard_normal(100000).astype(np.float32)
        self._n01_1e6 = self._rng.standard_normal(1000000).astype(np.float32)
        super(TestEvidenceLowerBound, self).setUp()
        device = paddle.set_device('gpu')
        paddle.disable_static(device)

    def test_objective(self):
        logqx = stats.norm.logpdf(self._n01_1e5).astype(np.float32)
        qx_samples = paddle.to_tensor(self._n01_1e5)
        logqx = paddle.to_tensor(logqx)

        def _check_elbo(x_mean, x_std):
            _x_mean = paddle.to_tensor(x_mean, stop_gradient=True)
            _x_std = paddle.to_tensor(x_std, stop_gradient=True)
            model = ELBO(TestNet_Gen(_x_mean, _x_std), TestNet_Var(qx_samples, logqx))
            lower_bound = -model({}).numpy()
            analytic_lower_bound = -_kl_normal_normal(paddle.to_tensor(0.), paddle.to_tensor(1.), _x_mean, _x_std)\
                .numpy()
            # print(lower_bound, analytic_lower_bound)
            self.assertAlmostEqual(lower_bound[0], analytic_lower_bound[0], delta=1e-3)
        _check_elbo(0., 1.)
        _check_elbo(2., 3.)

    def test_sgvb(self):
        eps_samples = paddle.to_tensor(self._n01_1e5)
        mu = paddle.to_tensor(2., stop_gradient=False)
        sigma = paddle.to_tensor(3., stop_gradient=False)
        qx_samples = eps_samples * sigma + mu
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        def _check_sgvb(x_mean, x_std, atol=1e-6, rtol=1e-6):
            _x_mean = paddle.to_tensor(x_mean)
            _x_std = paddle.to_tensor(x_std)
            model = ELBO(TestNet_Gen(_x_mean, _x_std), TestNet_Var(qx_samples, log_qx), estimator='sgvb')
            sgvb_cost = model({})
            sgvb_grads = paddle.grad(sgvb_cost, [mu, sigma], retain_graph=True)
            sgvb_grads = paddle.concat(sgvb_grads).numpy()
            true_cost = _kl_normal_normal(mu, sigma, x_mean, x_std)
            true_grads = paddle.grad(true_cost, [mu, sigma], retain_graph=True)
            true_grads = paddle.concat(true_grads).numpy()
            print('sgvb_grads: ', sgvb_grads)
            print('true_grads: ', true_grads)
            np.testing.assert_allclose(sgvb_grads, true_grads, atol=atol, rtol=rtol)

        _check_sgvb(0., 1., rtol=1e-2)
        _check_sgvb(2., 3., atol=1e-2)


    def test_reinforce(self):
        eps_samples = paddle.to_tensor(self._n01_1e6)
        mu = paddle.to_tensor(2., stop_gradient=False)
        sigma = paddle.to_tensor(3., stop_gradient=False)
        qx_samples = eps_samples * sigma + mu
        qx_samples.stop_gradient = True
        norm = Normal(mean=mu, std=sigma)
        log_qx = norm.log_prob(qx_samples)

        def _check_reinforce(x_mean, x_std, atol=1e-6, rtol=1e-6):
            _x_mean = paddle.to_tensor(x_mean)
            _x_std = paddle.to_tensor(x_std)
            model = ELBO(TestNet_Gen(_x_mean, _x_std), TestNet_Var(qx_samples, log_qx), estimator='reinforce')
            #TODO: Check grads when use variance reduction and baseline
            reinforce_cost = model({}, variance_reduction=False)
            reinforce_grads = paddle.grad(reinforce_cost, [mu, sigma], retain_graph=True)
            reinforce_grads = paddle.concat(reinforce_grads).numpy()
            true_cost = _kl_normal_normal(mu, sigma, _x_mean, _x_std)
            true_grads = paddle.grad(true_cost, [mu, sigma], retain_graph=True)
            true_grads = paddle.concat(true_grads).numpy()
            print('reinforce_grads: ', reinforce_grads)
            print('true_grads: ', true_grads)
            np.testing.assert_allclose(reinforce_grads, true_grads, rtol=rtol, atol=atol)

        _check_reinforce(0., 1., rtol=1e-2)
        _check_reinforce(2., 3., atol=1e-6)

if __name__ == '__main__':
    unittest.main()
