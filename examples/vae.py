# Copyright TODO

""" VAE example code """

import os
import math
import gzip
import progressbar
import six
import numpy as np

from PIL import Image
from six.moves import urllib, range
from six.moves import cPickle as pickle

import paddle
import paddle.fluid as fluid
from paddle.io import Dataset
from paddle.nn import functional as F

from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO

from utils import load_mnist_realval, save_img, MNISTDataset

device = paddle.set_device('gpu') # or 'gpu'
paddle.disable_static(device)

class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = paddle.nn.Linear(z_dim, 500)
        self.act1 = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(500, 500)
        self.act2 = paddle.nn.ReLU()

        self.fc2_ = paddle.nn.Linear(500, 28*28)
        self.act2_ = paddle.nn.Sigmoid()

    def forward(self, observed):
        # Check mark, mean, std, shape
        try:
            batch_len = observed['z'].shape[0]
        except:
            batch_len = 100

        sd = fluid.layers.ones(shape=(batch_len, self.z_dim), dtype='float32')
        mean = fluid.layers.zeros(shape=(batch_len, self.z_dim), dtype='float32')

        z = self.Normal('z', mean=mean,
                            std=sd,
                            shape=(()), 
                            observation=observed,
                            reparameterize=False)

        x_probs = self.act2_(self.fc2_(self.act2(self.fc2(self.act1(self.fc1(z))))))

        self.cache['x_mean'] = (x_probs, 0,)

        sample_x = self.Bernoulli('x', shape=(()), 
                                  observation=observed,
                                  probs=x_probs)

        assert(sample_x.shape[0] == batch_len)
        return self


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.fc1 = paddle.nn.Linear(x_dim, 500)
        self.act1 = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(500, 500)
        self.act2 = paddle.nn.ReLU()

        self.fc3 = paddle.nn.Linear(500, z_dim)
        self.fc4 = paddle.nn.Linear(500, z_dim)

    def forward(self, observed):
        x = observed['x']
        z_logits = self.act2(self.fc2(self.act1(self.fc1(x))))

        z_mean = self.fc3(z_logits)
        z_sd = paddle.exp(self.fc4(z_logits))

        z= self.Normal('z', mean=z_mean, std=z_sd, 
                            shape=(()), 
                            observation=observed,
                            reparameterize=True)
        
        assert( z.shape[1] == self.z_dim)
        return self


class ReduceMeanLoss(paddle.nn.Layer):
    def __init__(self):
        super(ReduceMeanLoss, self).__init__()

    def forward(self, input_, label):
        return input_

def main():

    epoch_size = 10 
    batch_size = 64

    # Define model parameters
    z_dim = 40
    x_dim = 28*28*1

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    network = ELBO(generator, variational)

    # define loss
    lr = 0.001
    net_loss = ReduceMeanLoss()

    # define the optimizer
    net_opt = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=network.parameters())

    # Model training
    model = paddle.Model(network)
    model.prepare(net_opt, net_loss)

    model.fit(train_data=MNISTDataset(mode='train'), # train_dataset, #MNISTDataset(mode='train'),
              eval_data=MNISTDataset(mode='valid'), # test_dataset,  #MNISTDataset(mode='valid'),
              batch_size=batch_size,
              epochs=epoch_size,
              eval_freq=10,
              save_dir='../model/',
              save_freq=10,
              verbose=1,
              shuffle=True,
              num_workers=0,
              callbacks=None)

    # Model test
    test_data = MNISTDataset(mode='test')
    batch_x = test_data.data[0:batch_size]

    # print(len(model.parameters()))
    # params_info = model.summary()

    # generator, variational = list(model.network.children())
    # print([len(generator.parameters()), len(variational.parameters())])

    nodes_q = variational({'x': paddle.to_tensor(batch_x)}).nodes
    z, _ = nodes_q['z']
    cache = generator({'z': z}).cache
    sample = cache['x_mean'][0].numpy()

    if not os.path.exists('result'):
        os.mkdir('result')
    print([sample.shape, batch_x.shape])

    cache = generator({}).cache
    sample_gen = cache['x_mean'][0].numpy()

    result_fold = 'result'
    if not os.path.isdir(result_fold):
        os.mkdir(result_fold)

    save_img(batch_x, os.path.join(result_fold, 'origin_x_.png' ))
    save_img(sample,  os.path.join(result_fold, 'reconstruct_x_.png' ))
    save_img(sample_gen,  os.path.join(result_fold, 'sample_x_.png' ))


if __name__ == '__main__':
    main()
