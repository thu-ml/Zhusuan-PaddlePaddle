# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import paddle
from scipy import stats
import numpy as np
import six

from zhusuan import mcmc
from zhusuan.framework import BayesianNet

import unittest


def sample_error_with(sampler, n_chains=1, n_iters=80000, thinning=50, burinin=None, dtype='float32'):
    if burinin is None:
        burinin = n_iters * 2 // 3

    class Test_Model(BayesianNet):
        def __init__(self):
            super().__init__()
            self.x = None

        def forward(self, observed):
            self.observe(observed)
            sample = self.sn('Normal',
                             name='x',
                             mean=paddle.zeros([n_chains], dtype=dtype),
                             std=2)
            sample.stop_gradient = False
            return self

        def log_joint(self, use_cache=False):
            x = self.observed['x']
            lh_noise = paddle.normal(shape=paddle.shape(self.observed['x']), std=2.)
            return 2 * (x ** 2) - x ** 4 + lh_noise

    x = paddle.zeros([n_chains], dtype=dtype)
    model = Test_Model()
    samples = []
    for t in range(n_iters):
        x_sample = sampler.sample(model, {}, {'x': x})['x'].numpy()
        if np.isnan(x_sample.sum()):
            raise ValueError("nan encountered")
        if t >= burinin and t % thinning == 0:
            samples.append(x_sample)
    samples = np.array(samples)
    samples = samples.reshape(-1)
    A = 3
    xs = np.linspace(-A, A, 1000)
    pdfs = np.exp(2*(xs**2) - xs**4)
    pdfs = pdfs / pdfs.mean() / A / 2
    est_pdfs = stats.gaussian_kde(samples)(xs)
    return np.abs(est_pdfs - pdfs).mean()


class TestMCMC(unittest.TestCase):

    def test_hmc(self):
        sampler = mcmc.HMC(step_size=0.01, n_leapfrogs=10)
        e = sample_error_with(sampler, n_chains=100, n_iters=1000)
        print(e)

class TestSGMCMC(unittest.TestCase):

    def test_sgld(self):
        sampler = mcmc.SGLD(learning_rate=0.01, iters=1)
        e = sample_error_with(sampler, n_chains=100, n_iters=8000)
        print(e)
