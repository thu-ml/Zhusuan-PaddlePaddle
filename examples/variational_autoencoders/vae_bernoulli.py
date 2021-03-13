# Copyright TODO

"""VAE example code"""
import sys
import os
import math

import paddle
import paddle.fluid as fluid

sys.path.append('..')
from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO

from utils import load_mnist_realval, save_img, standardize

device = paddle.set_device('gpu')
paddle.disable_static(device)


class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = paddle.nn.Linear(z_dim, 500, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm(500, act='relu')
        self.fc2 = paddle.nn.Linear(500, 500, bias_attr=False)
        self.bn2 = paddle.nn.BatchNorm(500, act='relu')
        self.fc3 = paddle.nn.Linear(500, x_dim)
        # self.act3 = paddle.nn.Sigmoid()

    def forward(self, observed):
        self.observe(observed)
        try:
            batch_len = self.observed['z'].shape[0]
        except:
            batch_len = 100
        z_probs = fluid.layers.zeros(shape=[batch_len, self.z_dim], dtype='float32')
        z = self.stochastic_node('Bernoulli',
                                 name='z',
                                 probs=z_probs,
                                 reparameterize=False,
                                 reduce_mean_dims=[0],
                                 reduce_sum_dims=[1])
        # x_probs = self.act3(self.fc3(self.bn2(self.fc2(self.bn1(self.fc1(z))))))
        x_probs = self.fc3(self.bn2(self.fc2(self.bn1(self.fc1(z)))))
        self.cache['x_mean'] = (paddle.nn.functional.sigmoid(x_probs), 0,)

        sample_x = self.stochastic_node('Bernoulli',
                                        name='x',
                                        probs=x_probs,
                                        reparameterize=False,
                                        reduce_mean_dims=[0],
                                        reduce_sum_dims=[1])
        return self


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = paddle.nn.Linear(x_dim, 500, bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm(500, act='relu')
        self.fc2 = paddle.nn.Linear(500, 500, bias_attr=False)
        self.bn2 = paddle.nn.BatchNorm(500, act='relu')
        self.fc3 = paddle.nn.Linear(500, z_dim)
        # self.act3_ = paddle.nn.Sigmoid()

    def forward(self, observed):
        self.observe(observed)
        x = self.observed['x']
        # TODO: Find out why z_probs == nan during train
        z_probs = self.fc3(self.bn2(self.fc2(self.bn1(self.fc1(x)))))
        z = self.stochastic_node('Bernoulli',
                                 name='z',
                                 probs=z_probs,
                                 reparameterize=True,
                                 reduce_mean_dims=[0],
                                 reduce_sum_dims=[1])
        return self


class Baseline(paddle.nn.Layer):
    def __init__(self, x_dim):
        super(Baseline, self).__init__()
        self.fc1 = paddle.nn.Linear(x_dim, 100)
        self.act1 = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(100, 1)

    def forward(self, x):
        lc_x = self.fc2(self.act1(self.fc1(x)))
        lc_x = fluid.layers.reduce_mean(lc_x)
        lc_x = fluid.layers.squeeze(lc_x, [0])
        return lc_x


def main():
    # Define model parameters
    epoch_size = 10
    batch_size = 64

    z_dim = 40
    x_dim = 28 * 28 * 1

    lr = 0.0001

    # Create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    baseline_net = Baseline(x_dim)
    model = ELBO(generator, variational, estimator='sgvb')

    clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                      parameters=model.parameters(), grad_clip=clip)

    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval()
    # do train
    len_ = x_train.shape[0]
    num_batches = math.ceil(len_ / batch_size)
    model.train()
    for epoch in range(epoch_size):
        for step in range(num_batches):
            x = paddle.to_tensor(x_train[step * batch_size:min((step + 1) * batch_size, len_)])
            x = paddle.reshape(x, [-1, x_dim])
            # cx = baseline_net(x)
            # cost, baseline_cost = model({'x': x}, baseline=cx)
            # loss = cost + baseline_cost
            loss = model({'x': x})
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if (step + 1) % 100 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, epoch_size, step + 1, num_batches, float(loss.numpy())))

    # eval
    batch_x = x_test[0:batch_size]
    nodes_q = variational({'x': paddle.to_tensor(batch_x)}).nodes
    z = nodes_q['z'].tensor
    cache = generator({'z': z}).cache
    sample = cache['x_mean'][0].numpy()

    cache = generator({}).cache
    sample_gen = cache['x_mean'][0].numpy()

    result_fold = 'result'
    if not os.path.exists(result_fold):
        os.mkdir(result_fold)

    print([sample.shape, batch_x.shape])
    save_img(batch_x, os.path.join(result_fold, 'origin_x_bernoulli_reinforce.png'))
    save_img(sample, os.path.join(result_fold, 'reconstruct_x_bernoulli_reinforce.png'))
    save_img(sample_gen, os.path.join(result_fold, 'sample_x_bernoulli_reinforce.png'))


if __name__ == '__main__':
    main()
