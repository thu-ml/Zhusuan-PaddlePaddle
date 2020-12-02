# Copyright TODO

""" VAE example code """
import sys
import os
import math

import paddle
import paddle.fluid as fluid

sys.path.append('..')
from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO

from utils import load_mnist_realval, save_img


device = paddle.set_device('gpu')
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
        self.observe(observed)
        try:
            batch_len = self.observed['z'].shape[0]
        except:
            batch_len = 100

        std = fluid.layers.ones(shape=(batch_len, self.z_dim), dtype='float32')
        mean = fluid.layers.zeros(shape=(batch_len, self.z_dim), dtype='float32')

        z = self.stochastic_node('Normal',
                                 name='z', 
                                 mean=mean,
                                 std=std,
                                 shape=(()), 
                                 reparameterize=False)
        #print(z)

        x_probs = self.act2_(self.fc2_(self.act2(self.fc2(self.act1(self.fc1(z))))))

        self.cache['x_mean'] = (x_probs, 0,)

        sample_x = self.sn('Bernoulli',
                           name='x', 
                           shape=(()), 
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
        self.observe(observed)
        x = self.observed['x']
        z_logits = self.act2(self.fc2(self.act1(self.fc1(x))))

        z_mean = self.fc3(z_logits)
        z_sd = paddle.exp(self.fc4(z_logits))

        z = self.stochastic_node('Normal',
                                 name='z', 
                                 mean=z_mean, 
                                 std=z_sd, 
                                 shape=(()), 
                                 reparameterize=True)
        
        assert( z.shape[1] == self.z_dim)
        return self

def main():

    # Define model parameters
    epoch_size = 10
    batch_size = 64

    z_dim = 40
    x_dim = 28*28*1

    lr = 0.001

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    model = ELBO(generator, variational)

    clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                      parameters=model.parameters(),)

    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval()

    # do train
    len_ = x_train.shape[0]
    num_batches = math.ceil(len_ / batch_size)

    for epoch in range(epoch_size):
        for step in range(num_batches):
            x = paddle.to_tensor(x_train[step*batch_size:min((step+1)*batch_size, len_)])
            x = paddle.reshape(x,[-1, x_dim])

            ##loss= model(x)
            loss= model({'x':x})
            assert(generator.log_joint().shape[0] == x.shape[0])

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
    save_img(batch_x, os.path.join(result_fold, 'origin_x_.png' ))
    save_img(sample,  os.path.join(result_fold, 'reconstruct_x_.png' ))
    save_img(sample_gen,  os.path.join(result_fold, 'sample_x_.png' ))


if __name__ == '__main__':
    main()
