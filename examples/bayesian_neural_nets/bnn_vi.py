# Copyright TODO

""" BNN with VI.ELBO  example code """
import sys
import os
import math
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable

sys.path.append('..')
import conf

from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO

from utils import load_uci_boston_housing, save_img, standardize

device = paddle.set_device('gpu')
paddle.disable_static(device)

class Net(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles
        self.y_logstd = self.create_parameter(shape=[1], dtype='float32')

    def forward(self, observed):
        self.observe(observed)
        
        x = self.observed['x']
        h = paddle.tile(x, [self.n_particles, *len(x.shape)*[1]])

        batch_size = x.shape[0]

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w = self.sn('Normal',
                        name="w" + str(i), 
                        mean=fluid.layers.zeros(shape=[n_out, n_in + 1], dtype='float32'), 
                        std=fluid.layers.ones(shape=[n_out, n_in +1], dtype='float32'),
                        group_ndims=2, 
                        n_samples=self.n_particles)

            w = fluid.layers.unsqueeze(w, axes=[1])
            w = paddle.tile(w, [1, batch_size, 1,1])
            h = paddle.concat([h, fluid.layers.ones(shape=[*h.shape[:-1], 1], dtype='float32')], -1)
            h = paddle.reshape(h, h.shape + [1])
            #print('w.shape: ', w.shape)
            #print('h.shape: ', h.shape)
            p = fluid.layers.sqrt(paddle.to_tensor(h.shape[2], dtype='float32'))
            #print('p.shape: ', p.shape)
            h = paddle.matmul(w, h)  / p
            h = paddle.squeeze(h, [-1])
            #print('w*h.shape: ', h.shape)

            if i < len(self.layer_sizes) - 2:
                h = paddle.nn.ReLU()(h)

        y_mean = fluid.layers.squeeze(h, [2])
        #print('y_mean.shape: ', y_mean.shape)
        y = self.observed['y']
        y_pred = fluid.layers.reduce_mean(y_mean,[0])
        self.cache['rmse'] = fluid.layers.sqrt(fluid.layers.reduce_mean((y - y_pred)**2))

        self.sn('Normal',
                name='y',
                mean=y_mean,
                logstd=self.y_logstd,
                reparameterize=True)

        return self


class Variational(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles

        self.w_means = [] 
        self.w_logstds = [] 

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w_mean_ = self.create_parameter(shape=[n_out, n_in+1], dtype='float32', is_bias=True)
            self.w_means.append(w_mean_)
            w_logstd_ = self.create_parameter(shape=(n_out, n_in +1),dtype='float32')
            self.w_logstds.append(w_logstd_)

        self.w_means = paddle.nn.ParameterList(self.w_means)
        self.w_logstds = paddle.nn.ParameterList(self.w_logstds)

    def forward(self, observed):
        self.observe(observed)
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            self.sn('Normal',
                    name='w' + str(i),
                    mean=self.w_means[i],
                    logstd=self.w_logstds[i],
                    group_ndims=2,
                    n_samples=self.n_particles,
                    reparametrize=True)
        return self

def main():
    # Load UCI Boston housing data
    data_path = os.path.join(conf.data_dir, "housing.data")
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        load_uci_boston_housing(data_path)
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    n_train, x_dim = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = standardize(
        y_train, y_test)
    
    print('data size: ', len(x_train))

    # Define model parameters
    lb_samples = 512
    epoch_size = 5000
    batch_size = 114

    n_hiddens = [50]
    layer_sizes = [x_dim] + n_hiddens + [1]
    print('layer size: ', layer_sizes)

    # create the network
    net = Net(layer_sizes, lb_samples)
    variational = Variational(layer_sizes, lb_samples)

    model = ELBO(net, variational)
    lr = 0.001
    clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                      parameters=model.parameters(),)
    
    print('parameters length: ', len(model.parameters()))
    # do train
    len_ = len(x_train)
    num_batches = math.floor(len_ / batch_size)

    # Define training/evaluation parameters
    test_freq = 20

    for epoch in range(epoch_size):
        perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[perm, :]
        y_train = y_train[perm]

        for step in range(num_batches):
            x = paddle.to_tensor(x_train[step*batch_size:(step+1)*batch_size])
            y = paddle.to_tensor(y_train[step*batch_size:(step+1)*batch_size])

            lbs = model({'x':x, 'y':y})
            lbs.backward()
            optimizer.step()
            optimizer.clear_grad()

            if (step + 1) % num_batches == 0:
                rmse = net.cache['rmse'].numpy()
                print("Epoch[{}/{}], Step [{}/{}], Lower bound: {:.4f}, RMSE: {:.4f}"
                      .format(epoch + 1, epoch_size, step + 1, num_batches, float(lbs.numpy()), float(rmse )* std_y_train))

        # eval
        if epoch % test_freq == 0:
            x_t = paddle.to_tensor(x_test)
            y_t = paddle.to_tensor(y_test)
            lbs = model({'x':x_t, 'y':y_t})
            rmse = net.cache['rmse'].numpy()
            print('>> TEST')
            print('>> Test Lower bound: {:.4f}, RMSE: {:.4f}'.format(float(lbs.numpy()), float(rmse) * std_y_train))


if __name__ == '__main__':
    main()
