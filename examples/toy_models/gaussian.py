#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.append('..')

import zhusuan as zs
from zhusuan.framework.bn import BayesianNet
from zhusuan import mcmc

import paddle
import paddle.fluid as fluid

class Gaussian(BayesianNet):
    def __init__(self, n_x, std, n_particles):
        super().__init__()

        self._n_x = n_x
        self._std = paddle.to_tensor(std)
        self._n_particles = n_particles

    def forward(self, observed):
        self.observe(observed)

        sample = self.sn('Normal',
                         name='w',
                         mean=fluid.layers.zeros(shape=[self._n_x],dtype='float32'), 
                         std=self._std, 
                         n_samples=self._n_particles)

        sample.stop_gradient = False
        
        return sample


if __name__ == "__main__":

    #n_x = 5
    n_x = 1
    std = 1 / (np.arange(n_x, dtype=np.float32) + 1)

    # Define HMC parameters
    kernel_width = 0.1
    n_chains = 500
    n_iters = 10
    burnin = n_iters // 2
    n_leapfrogs = 1000

    # Build the computation graph
    model = Gaussian(n_x, std, n_chains)

    sampler_type = 'HMC'
    if sampler_type == "SGLD":
        sampler = mcmc.SGLD(learning_rate=1e-3)
    else: # HMC
        sampler = mcmc.HMC(step_size=1e-2, n_leapfrogs=n_leapfrogs)

    samples = []
    time_st = time.time()
    print('Sampling...')
    for i in range(n_iters):
        if i % 2 == 0:
            d_time = time.time() - time_st
            time_st = time.time()
            print('step: {}, time: {:4f}s'.format(i, d_time))

        if sampler_type == "SGLD":
            resample = True if i == 0 else False
            sample_ = sampler.sample(model, {}, resample, step=10000)
        else:
            ip = {'w': 1.0 * fluid.layers.ones(shape=[n_chains, n_x],dtype='float32')}
            sample_ = sampler.sample(model, {}, initial_position=ip)

        samples.append(sample_['w'].numpy())

    print('Finished.')
    samples = np.vstack(samples)

    # Check & plot the results
    print('Expected mean = {}'.format(np.zeros(n_x)))
    print('Sample mean = {}'.format(np.mean(samples, 0)))
    print('Expected stdev = {}'.format(std))
    print('Sample stdev = {}'.format(np.std(samples, 0)))
    print('Relative error of stdev = {}'.format(
        (np.std(samples, 0) - std) / std))
