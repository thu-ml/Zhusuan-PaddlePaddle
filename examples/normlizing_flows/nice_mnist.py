import math
import os
import time

import paddle
import paddle.io
import numpy as np

from zhusuan.transforms.invertible import *
from zhusuan.framework import BayesianNet

from examples.utils import fetch_dataloaders, save_image


device = paddle.set_device('gpu')
paddle.disable_static(device)


class NICE(BayesianNet):
    def __init__(self, num_coupling, in_out_dim, mid_dim, hidden, mask_config):
        super().__init__()
        self.in_out_dim = in_out_dim
        masks = get_coupling_mask(in_out_dim, 1, num_coupling)
        coupling_layer = [
            AdditiveCoupling(in_out_dim=in_out_dim,
                             mid_dim=mid_dim,
                             hidden=hidden,
                             mask=masks[i])
            for i in range(num_coupling)]
        scaling_layer = Scaling(in_out_dim)
        self.flow = Sequential(coupling_layer + [scaling_layer])
        self.sn('Logistic',
                name='prior_x',
                loc=0.,
                scale=1.)
        self.sn('FlowDistribution',
                name='x',
                latents=self.nodes['prior_x'].dist,
                transformation=self.flow,
                n_samples=-1)  # Not sample when initializing

    def sample(self, size):
        return self.nodes['x'].dist.sample(shape=[size, self.in_out_dim])

    def forward(self, x):
        return self.nodes['x'].log_prob(x)


def main():
    batch_size = 200
    epoch_size = 1
    sample_size = 64
    coupling = 4
    mask_config = 1.

    # Optim Parameters
    lr = 1e-3

    full_dim = 1 * 28 * 28
    mid_dim = 1000
    hidden = 5

    model = NICE(num_coupling=coupling,
                 in_out_dim=full_dim,
                 mid_dim=mid_dim,
                 hidden=hidden,
                 mask_config=mask_config)

    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters(), epsilon=1e-4)

    train_dataloader, test_dataloader = fetch_dataloaders('MNIST', batch_size, logit_transform=False, dequantify=True)

    for epoch in range(epoch_size):
        stats = []
        for _, data in enumerate(train_dataloader()):
            model.train()
            optimizer.clear_grad()
            inputs = data[0]
            loss = -model.nodes['x'].log_prob(inputs).mean()
            loss.backward()
            optimizer.step()
            stats.append(loss.numpy())
        print("Epoch:[{}/{}], Log Likelihood: {:.4f}".format(
            epoch + 1, epoch_size, np.mean(np.array(stats))
        ))

    model.eval()
    with paddle.no_grad():
        samples = model.nodes['x'].dist.sample(shape=[sample_size, full_dim])
        samples = paddle.reshape(samples, shape=[-1, 1, 28, 28])
        save_image(samples, './results/NICE/sample-NICE-1.png')


if __name__ == '__main__':
    main()
