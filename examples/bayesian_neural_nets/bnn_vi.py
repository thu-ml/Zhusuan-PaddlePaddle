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
from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO
import conf

from utils import load_uci_boston_housing, save_img

device = paddle.set_device('gpu')
paddle.disable_static(device)

class Net(BayesianNet):
    def __init__(self, layer_sizes, n_particles, batch_size):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles
        self.batch_size = batch_size

        self.y_logstd = self.create_parameter(shape=[1], dtype='float32')

    def forward(self, observed):
        """
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
            w = paddle.tile(w, [1, self.batch_size, 1,1])
            h = paddle.concat([h, fluid.layers.ones(shape=[*h.shape[:-1], 1], dtype='float32')], -1)
            h = paddle.reshape(h, h.shape + [1])
            print('w.shape: ', w.shape)
            print('h.shape: ', h.shape)
            #h = paddle.matmul(w, h) / fluid.layers.sqrt(paddle.cast(h.shape[2], 'float32'))
            p = fluid.layers.sqrt(paddle.to_tensor(h.shape[2], dtype='float32'))
            print('p.shape: ', p.shape)
            h = paddle.matmul(w, h)  / p
            h = paddle.squeeze(h, [-1])
            print('w*h.shape: ', h.shape)

            if i < len(self.layer_sizes) - 2:
                #h = self.acts[i](h)
                h = paddle.nn.ReLU()(h)

        y_mean = fluid.layers.squeeze(h, [2])
        print('y_mean.shape: ', y_mean.shape)

        sample = self.sn('Normal',
                name='y',
                mean=y_mean,
                logstd=self.y_logstd,
                reparameterize=True)
        print('sample.shape: ', sample.shape)
        print('y_logstd.shape: ', self.y_logstd.shape)

        #return self

        #sample = y_mean * self.y_logstd
        sample = paddle.normal(name='1',
                               shape=(32,10))
        """

        sample = paddle.tile(self.y_logstd, [1,1])
        #sample = paddle.tile(self.y_logstd, [2,1])
        return sample


class Variational(BayesianNet):
    def __init__(self, layer_sizes, n_particles, batch_size):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_particles = n_particles
        self.batch_size = batch_size

        self.w_means = [] 
        self.w_logstds = [] 

        #self.fc = self.create_parameter(shape=[1, 2], dtype='float32')
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            fc = self.create_parameter(shape=[n_out, n_in+1], dtype='float32')
            print('fc.shape: ', fc.shape)
            self.w_means.append(fc)
            fc = self.create_parameter(shape=(n_out, n_in +1),dtype='float32')
            self.w_logstds.append(fc)
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
    
    print('data size: ', len(x_train))

    # Define model parameters
    lb_samples = 32
    ll_samples = 5000
    epoch_size = 500
    batch_size = 10

    n_hiddens = [50]
    layer_sizes = [x_dim] + n_hiddens + [1]
    print('layer size: ', layer_sizes)

    # create the network
    net = Net(layer_sizes, lb_samples, batch_size)
    #net.forward(observed={'x':fluid.layers.ones(shape=(batch_size, x_dim), dtype='float32')})
    #print(net.parameters())
    #print(paddle.summary(net))
    variational = Variational(layer_sizes, lb_samples, batch_size)
    #variational.forward({})
    #print(variational.parameters())

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
    test_freq = 10

    for epoch in range(epoch_size):
        for step in range(num_batches):
            x = paddle.to_tensor(x_train[step*batch_size:(step+1)*batch_size])
            print('x.shape: ', x.shape)

            ##loss= model(x)
            #loss= model({'x':x})
            #loss= fluid.layers.reduce_mean(net({'x':x}).nodes['y'].tensor)
            loss = fluid.layers.reduce_mean(net({'x':x}))
            #loss = fluid.layers.reduce_mean(variational({}).nodes['w1'].tensor)

            print('loss: ', loss)
            loss.backward()
            print(net.y_logstd.grad)
            optimizer.step()
            optimizer.clear_grad()

            if (step + 1) % 100 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, epoch_size, step + 1, num_batches, float(loss.numpy())))

        # eval
        if epoch % test_freq == 0:
            pass


if __name__ == '__main__':
    main()
