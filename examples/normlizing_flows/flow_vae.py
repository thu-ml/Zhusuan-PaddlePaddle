import sys
import os
import math

import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Assign, Constant, Uniform

from zhusuan.framework.bn import BayesianNet
from zhusuan.variational.elbo import ELBO
from zhusuan.transforms.invertible import *
from examples.utils import load_mnist_realval, save_img

device = paddle.set_device('gpu')
paddle.disable_static(device)


class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        # Build Decoder
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = paddle.nn.Linear(z_dim, 500)
        self.act1 = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(500, 500)
        self.act2 = paddle.nn.ReLU()

        self.fc2_ = paddle.nn.Linear(500, 28 * 28)
        # self.act2_ = paddle.nn.Sigmoid()

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
                                 reparameterize=False,
                                 reduce_mean_dims=[0],
                                 reduce_sum_dims=[1])
        # print(z)

        # x_probs = self.act2_(self.fc2_(self.act2(self.fc2(self.act1(self.fc1(z))))))
        x_probs = self.fc2_(self.act2(self.fc2(self.act1(self.fc1(z)))))
        self.cache['x_mean'] = (paddle.nn.functional.sigmoid(x_probs), 0,)

        sample_x = self.sn('Bernoulli',
                           name='x',
                           shape=(()),
                           probs=x_probs,
                           reduce_mean_dims=[0],
                           reduce_sum_dims=[1])

        assert (sample_x.shape[0] == batch_len)
        return self


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        # Build Encoder
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

        # Save z_logits for householder flow
        self.cache['z_logits'] = z_logits

        z_mean = self.fc3(z_logits)
        z_sd = paddle.exp(self.fc4(z_logits))

        z = self.stochastic_node('Normal',
                                 name='z',
                                 mean=z_mean,
                                 std=z_sd,
                                 shape=(()),
                                 reparameterize=True,
                                 reduce_mean_dims=[0],
                                 reduce_sum_dims=[1])

        assert (z.shape[1] == self.z_dim)
        return self


class NICEFlow(nn.Layer):
    def __init__(self, z_dim, mid_dim, num_coupling, num_hidden):
        super().__init__()
        masks = get_coupling_mask(z_dim, 1, num_coupling)
        coupling_layer = [
            AdditiveCoupling(in_out_dim=z_dim,
                             mid_dim=mid_dim,
                             hidden=num_hidden,
                             mask=masks[i])
            for i in range(num_coupling)]
        scaling_layer = Scaling(z_dim)
        self.flow = Sequential(coupling_layer + [scaling_layer])

    def forward(self, z, **kwargs):
        out, log_det = self.flow.forward(z, **kwargs)
        res = {'z': out[0]}
        return res, log_det


class HouseHolderFlow(nn.Layer):
    def __init__(self, z_dim, v_dim, n_flows):
        super(HouseHolderFlow, self).__init__()

        class HF(nn.Layer):
            def __init__(self, z_dim, is_first=False, v_dim=None):
                super(HF, self).__init__()
                stdv = 1. / math.sqrt(z_dim)
                if is_first:
                    self.v_layer = nn.Linear(v_dim, z_dim, weight_attr=Uniform(-stdv, stdv),
                                             bias_attr=Uniform(-stdv, stdv))
                else:
                    self.v_layer = nn.Linear(z_dim, z_dim, weight_attr=Uniform(-stdv, stdv),
                                             bias_attr=Uniform(-stdv, stdv))

            def forward(self, z, v, **kwargs):
                v_new = self.v_layer(v)
                vvT = paddle.bmm(v_new.unsqueeze(2), v_new.unsqueeze(1))
                vvTz = paddle.bmm(vvT, z.unsqueeze(2)).squeeze(2)
                norm_sq = paddle.sum(v_new * v_new, axis=1, keepdim=True)
                norm_sq = norm_sq.expand(shape=[norm_sq.shape[0], v_new.shape[1]])
                z_new = z - 2 * vvTz / norm_sq
                return (z_new, v_new), paddle.zeros(shape=[1, 1])

        _first_kwarg = {'is_first': True, 'v_dim': v_dim}
        self.flow = Sequential([HF(z_dim, **_first_kwarg) if _ == 0 else HF(z_dim) for _ in range(n_flows)])

    def forward(self, z, v, **kwargs):
        out, log_det = self.flow.forward(z, v, **kwargs)
        res = {'z': out[0]}
        return res, log_det


class PlanarFlow(nn.Layer):
    def __init__(self, z_dim, n_flows):
        super().__init__()

        class PF(nn.Layer):
            def __init__(self, z_dim):
                super(PF, self).__init__()
                self.u = self.create_parameter(shape=[1, z_dim],
                                               default_initializer=Assign(paddle.randn(shape=[1, z_dim])))
                self.w = self.create_parameter(shape=[1, z_dim],
                                               default_initializer=Assign(paddle.randn(shape=[1, z_dim])))
                self.b = self.create_parameter(shape=[1],
                                               default_initializer=Assign(paddle.randn(shape=[1])))
                self.add_parameter('u', self.u)
                self.add_parameter('w', self.w)
                self.add_parameter('b', self.b)

            def forward(self, z, **kwargs):
                def m(z):
                    return F.softplus(z) - 1.

                def h(z):
                    return paddle.tanh(z)

                def h_prime(z):
                    return 1. - paddle.square(h(z))

                inner = paddle.sum(self.w * self.u)
                u = self.u + (m(inner) - inner) * self.w / paddle.square(self.w.norm())
                activation = paddle.sum(self.w * z, axis=1, keepdim=True) + self.b
                z_new = z + u * h(activation)
                psi = h_prime(activation) * self.w
                log_det = paddle.log(paddle.abs(1. + paddle.sum(u * psi, axis=1, keepdim=True)))
                return (z_new,), log_det

        self.flows = Sequential([PF(z_dim) for _ in range(n_flows)])

    def forward(self, z, **kwargs):
        out, log_det = self.flows.forward(z, **kwargs)
        res = {'z': out[0]}
        return res, log_det


def main():
    # Define model parameters
    epoch_size = 10
    batch_size = 64

    z_dim = 40
    x_dim = 28 * 28 * 1

    mid_dim_flow = 64
    num_coupling = 10
    num_hidden_per_coupling = 4

    lr = 0.001

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    nice_flow = NICEFlow(z_dim, mid_dim_flow, num_coupling, num_hidden_per_coupling)
    model = ELBO(generator, variational, transform=nice_flow, transform_var=['z'])
    # planar_flow = PlanarFlow(z_dim, 1)
    # model = ELBO(generator, variational, transform=planar_flow, transform_var=['z'])
    # householder_flow = HouseHolderFlow(z_dim, 500, 5)
    # model = ELBO(generator, variational, transform=householder_flow, transform_var=['z'], auxillary_var=['z_logits'])
    # model = ELBO(generator, variational)

    clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                      parameters=model.parameters(), )

    x_train, t_train, x_valid, t_valid, x_test, t_test = load_mnist_realval()

    # do train
    len_ = x_train.shape[0]
    num_batches = math.ceil(len_ / batch_size)

    for epoch in range(epoch_size):
        for step in range(num_batches):
            x = paddle.to_tensor(x_train[step * batch_size:min((step + 1) * batch_size, len_)])
            x = paddle.reshape(x, [-1, x_dim])
            if x.shape[0] != batch_size:
                continue

            ##loss= model(x)
            loss = model({'x': x})
            assert (generator.log_joint().shape == [])

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

    z = paddle.to_tensor(np.load('z.npy').astype(np.float32))
    cache = generator({'z': z}).cache
    sample_gen = cache['x_mean'][0].numpy()

    result_fold = './results/flow-VAE'
    if not os.path.exists(result_fold):
        os.mkdir(result_fold)

    print([sample.shape, batch_x.shape])
    save_img(batch_x, os.path.join(result_fold, 'origin_x_VAE+NICE.png'))
    save_img(sample, os.path.join(result_fold, 'reconstruct_x_VAE+NICE.png'))
    save_img(sample_gen, os.path.join(result_fold, 'sample_x_VAE+NICE.png'))


if __name__ == '__main__':
    main()
