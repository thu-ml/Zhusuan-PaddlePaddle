# Copyright TODO

""" BNN with MCMC.SGLD  example code """
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
from zhusuan import mcmc

from utils import load_uci_boston_housing, save_img, standardize

device = paddle.set_device('gpu')
paddle.disable_static(device)

class Net(BayesianNet):
    def __init__(self, layer_sizes, n_particles):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles
        self.y_logstd = paddle.to_tensor(-1.95, dtype='float32')

        self.w_logstds = [] 

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w_logstd_ = self.create_parameter(shape=(n_out, n_in +1),dtype='float32')
            self.w_logstds.append(w_logstd_)

        self.w_logstds = paddle.nn.ParameterList(self.w_logstds)


    def forward(self, observed):
        self.observe(observed)
        
        x = self.observed['x']
        h = paddle.tile(x, [self.n_particles, *len(x.shape)*[1]])

        batch_size = x.shape[0]

        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w = self.sn('Normal',
                        name="w" + str(i), 
                        mean=fluid.layers.zeros(shape=[n_out, n_in + 1], dtype='float32'), 
                        logstd=self.w_logstds[i],
                        group_ndims=2, 
                        n_samples=self.n_particles,
                        reduce_mean_dims=[0],)

            w = fluid.layers.unsqueeze(w, axes=[1])
            w = paddle.tile(w, [1, batch_size, 1,1])
            h = paddle.concat([h, fluid.layers.ones(shape=[*h.shape[:-1], 1], dtype='float32')], -1)
            h = paddle.reshape(h, h.shape + [1])
            p = fluid.layers.sqrt(paddle.to_tensor(h.shape[2], dtype='float32'))
            h = paddle.matmul(w, h)  / p
            h = paddle.squeeze(h, [-1])

            if i < len(self.layer_sizes) - 2:
                h = paddle.nn.ReLU()(h)

        y_mean = fluid.layers.squeeze(h, [2])
        y = self.observed['y']
        y_pred = fluid.layers.reduce_mean(y_mean,[0])
        self.cache['rmse'] = fluid.layers.sqrt(fluid.layers.reduce_mean((y - y_pred)**2))

        self.sn('Normal',
                name='y',
                mean=y_mean,
                logstd=self.y_logstd,
                reparameterize=True,
                reduce_mean_dims=[0,1],
                multiplier=456,) ### training data size

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
    lb_samples = 20
    epoch_size = 5000
    batch_size = 114

    n_hiddens = [50]

    layer_sizes = [x_dim] + n_hiddens + [1]
    print('layer size: ', layer_sizes)

    # create the network
    net = Net(layer_sizes, lb_samples)
    print('parameters length: ', len(net.parameters()))

    lr = 1e-2
    model = mcmc.SGLD(lr)

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

            ## E-step 
            re_sample = True if epoch==0 and step ==0 else False
            w_samples = model.sample(net, {'x':x, 'y':y}, re_sample)

            ## M-step: update w_logstd
            for i,(k,w) in enumerate(w_samples.items()):
                assert(w.shape[0] == lb_samples)
                esti_logstd = 0.5 * paddle.log(fluid.layers.reduce_mean(w*w, [0]))
                net.w_logstds[i].set_value(esti_logstd)

            if (step + 1) % num_batches == 0:
                net.forward({**w_samples, 'x':x, 'y':y})
                rmse = net.cache['rmse'].numpy()
                print("Epoch[{}/{}], Step [{}/{}], RMSE: {:.4f}"
                      .format(epoch + 1, epoch_size, step + 1, num_batches, float(rmse )* std_y_train))

        # eval
        if epoch % test_freq == 0:
            x_t = paddle.to_tensor(x_test)
            y_t = paddle.to_tensor(y_test)
            net.forward({**w_samples, 'x':x_t, 'y':y_t})
            rmse = net.cache['rmse'].numpy()
            print('>> TEST')
            print('>> Test RMSE: {:.4f}'.format(float(rmse) * std_y_train))

if __name__ == '__main__':
    main()
