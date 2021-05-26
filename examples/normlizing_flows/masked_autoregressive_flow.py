import time

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn.initializer import Assign, Constant, Uniform

from zhusuan.framework import BayesianNet
from zhusuan.transforms.invertible import *
from examples.utils import fetch_dataloaders, save_image

import numpy as np
import os

device = paddle.set_device('gpu')
paddle.disable_static(device)


class MAF(BayesianNet):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size):
        super(MAF, self).__init__()
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules.append(MADE(input_size, hidden_size, n_hidden, cond_label_size, input_degrees=self.input_degrees))
            self.input_degrees = modules[-1].input_degrees.flip([0])
            modules.append(BatchNorm(input_size))
        self.flow = Sequential(modules)

        # prior distribution
        self.sn('Normal',
                name='prior_x',
                mean=paddle.zeros(shape=[input_size], dtype='float32'),
                std=paddle.ones(shape=[input_size], dtype='float32'),
                is_reparameterized=False, )
        self.sn('FlowDistribution',
                name='x',
                latents=self.nodes['prior_x'].dist,
                transformation=self.flow,
                n_samples=-1)  # Do not sample when model is initializing

    def forward(self, x, y):
        return self.nodes['x'].log_prob(x, cond_y=y)

    def sample(self, n_samples=1, cond_y=None):
        return self.nodes['x'].dist.sample(n_samples=n_samples, cond_y=cond_y)


def main():
    n_blocks = 5
    hidden_size = 512
    n_hidden = 1

    batch_size = 100
    n_epochs = 30
    lr = 1e-4

    train_dataloader, test_dataloader = fetch_dataloaders('MNIST', batch_size, logit_transform=True, dequantify=True)
    input_size = train_dataloader.dataset.input_size
    cond_label_size = train_dataloader.dataset.label_size

    model = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size)
    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters(), weight_decay=1e-6)

    model.train()
    for i in range(n_epochs):
        loss_cache = []
        # start = time.time()
        for _, data in enumerate(train_dataloader()):
            x, y = data
            x = x.reshape([x.shape[0], -1])
            loss = -model(x, y).mean()
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            loss_cache.append(loss.numpy())
        print('Epoch: {:3d} / {}, loss: {:.4f}'.format(i, n_epochs, np.mean(np.array(loss_cache))))
        # end = time.time()
        # print('Time: {}'.format(end - start))

    # Generate Sample
    lam = paddle.to_tensor(train_dataloader.dataset.lam)
    n_row = 8

    model.eval()
    samples = []
    labels = paddle.eye(cond_label_size)
    for i in range(cond_label_size):
        labels_i = labels[i].expand(shape=[n_row, -1])
        sample = model.sample(n_samples=n_row, cond_y=labels_i)
        log_probs = model(sample, labels_i).argsort(axis=0, descending=True).numpy()
        for l in log_probs:
            samples.append(sample[int(l)])
    samples = paddle.stack(samples)
    samples = paddle.reshape(samples, shape=[samples.shape[0], 1, 28, 28])
    samples = (paddle.nn.functional.sigmoid(samples) - lam) / (1 - 2 * lam)
    filename = 'generated_samples' + '_epoch{}'.format(n_epochs) + '.png'
    save_image(samples, os.path.join('./results/MAF', filename), nrow=n_row)


if __name__ == '__main__':
    main()
