
import os
import math
import gzip
import progressbar
import six
import numpy as np
from six.moves import urllib, range
from six.moves import cPickle as pickle
from PIL import Image

import paddle
import paddle.fluid as fluid
from paddle.io import Dataset
from paddle.nn import functional as F

from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO

from utils import load_mnist_realval, save_img

device = paddle.set_device('gpu') # or 'gpu'
paddle.disable_static(device)

pbar = None
examples_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(examples_dir, "data")

class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = paddle.nn.Linear(z_dim, 500)
        self.act1 = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(500, 500)
        self.act2 = paddle.nn.Sigmoid()

        self.fc2_ = paddle.nn.Linear(500, 28*28)

        self.fc3 = paddle.nn.Linear(500, x_dim)
        self.fc4 = paddle.nn.Linear(500, x_dim)

    def forward(self, observed):
        # Check mark, mean, std, shape
        batch_len = observed['z'].shape[0]
        sd = fluid.layers.ones(shape=(batch_len, self.z_dim), dtype='float32')
        mean = fluid.layers.zeros(shape=(batch_len, self.z_dim), dtype='float32')

        nodes = {}
        nodes = self.Normal('z', mean=mean,
                            std=sd,
                            shape=(()), observation=observed,
                            nodes=nodes, reparameterize=False)

        z = nodes['z'][0]
        x_logits = self.act2(self.fc2_(self.act1(self.fc1(z))))

        nodes['x_mean'] = (x_logits, 0,)
        nodes = self.Bernoulli('x', shape=(()), observation=observed,
                               nodes=nodes, probs=x_logits)
        return nodes


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
        nodes = {}
        z_logits = self.act2(self.fc2(self.act1(self.fc1(x))))

        z_mean = self.fc3(z_logits)
        z_sd = paddle.exp(self.fc4(z_logits))

        nodes = self.Normal('z', mean=z_mean, std=z_sd, shape=(()), \
                            observation=observed, nodes=nodes,
                            reparameterize=True)
        return nodes


class ReduceMeanLoss(paddle.nn.Layer):
    def __init__(self):
        super(ReduceMeanLoss, self).__init__()

    def forward(self, input_, label):
        return input_


class MyDataset(Dataset):
    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        # Load MNIST
        data_path = os.path.join(data_dir, "mnist.pkl.gz")
        x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval(data_path)
        if mode == 'train':
            self.data = x_train
            self.labels = t_train
        elif mode == 'test':
            self.data = x_test
            self.labels = t_test
        else:
            self.data = x_valid
            self.labels = t_valid

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

def main():

    epoch_size = 3000 #3000
    batch_size = 128

    # Define model parameters
    z_dim = 40
    x_dim = 28*28

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    network = ELBO(generator, variational)

    # define loss
    # learning rate setting
    lr = 0.001
    net_loss = ReduceMeanLoss()

    # define the optimizer
    net_opt = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=network.parameters())

    # Model training
    model = paddle.Model(network)
    model.prepare(net_opt, net_loss)

    model.fit(train_data=MyDataset(mode='train'), #train_dataset, #MyDataset(mode='train'),
              eval_data=MyDataset(mode='valid'), #test_dataset,  #MyDataset(mode='valid'),
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
    test_data = MyDataset(mode='test')
    batch_x = test_data.data[0:batch_size]

    # print(len(model.parameters()))
    # params_info = model.summary()

    generator, variational = list(model.network.children())
    # print([len(generator.parameters()), len(variational.parameters())])

    nodes_q = variational({'x': paddle.to_tensor(batch_x)})
    z, _ = nodes_q['z']
    nodes_p = generator({'z': z})

    sample = nodes_p['x_mean'][0].numpy()
    if not os.path.exists('result'):
        os.mkdir('result')
    print([sample.shape, batch_x.shape])

    # plt.figure(0)
    # cv2.imshow('input',batch_x[0])

    result_fold = 'result'
    if not os.path.isdir(result_fold):
        os.mkdir(result_fold)
    save_img(batch_x, os.path.join(result_fold, 'origin_x.png' ))
    save_img(sample,  os.path.join(result_fold, 'reconstruct_x.png' ))


if __name__ == '__main__':
    main()
