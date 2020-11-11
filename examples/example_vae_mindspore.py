import os 

#from mindspore import context
import mindspore.context as context
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore import Tensor

import mindspore as ms
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
import six

from PIL import Image
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
import mindspore._c_expression as me

context.set_context(save_graphs=True)


def save_img(data, name, size=32, num=32):
    """
    Visualize data and save to target files
    Args:
        data: nparray of size (num, size, size)
        name: ouput file name
        size: image size
        num: number of images
    """
    col = int(num / 8)
    row = 8

    imgs = Image.new('L', (size*col, size*row))
    for i in range(num):
        j = i/8
        img_data = data[i]
        img_data  = np.resize(img_data, (size, size))
        img_data = img_data * 255
        img_data = img_data.astype(np.uint8)
        im = Image.fromarray(img_data, 'L')
        imgs.paste(im, (int(j) * size , (i % 8) * size))
    imgs.save(name)


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """ create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    #mnist_ds = ds.MnistDataset(data_path)
    mnist_ds = ds.MnistDataset(data_path,num_samples=32)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # resize images to (32, 32)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)  # normalize images
    rescale_op = CV.Rescale(rescale, shift)  # rescale images
    hwc2chw_op = CV.HWC2CHW()  # change shape from (height, width, channel) to (channel, height, width) to fit network.
    type_cast_op = C.TypeCast(mstype.int32)  # change data type of label to int32 to fit network
    
    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class ReduceMeanLoss(nn.L1Loss):
    def construct(self, base, target):
        return base


class BayesianNet(nn.Cell):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.normal_dist = msd.Normal()
        self.fill = P.Fill()
        self.reduce_sum = P.ReduceSum(keep_dims=True)

    def normal(self, name, mean, sd, sample_shape, observed, nodes, reparameterize=True):
        if name in observed.keys():
            node = observed[name]
        else:
            if reparameterize:
                ones = self.fill(mstype.float32, sample_shape, 1.)
                zeros = self.fill(mstype.float32, sample_shape, 0.)
                epsilon = self.normal_dist('sample', sample_shape, zeros, ones)
                node = epsilon * sd + mean
            else:
                node = self.normal_dist('sample', sample_shape, mean, sd)
        logprob = self.reduce_sum(self.normal_dist('log_prob', node, mean, sd), 1)
        nodes[name] = (node, logprob)
        return nodes


class Generator(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = nn.Dense(z_dim, 500)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Dense(500, 500)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Dense(500, x_dim)
        self.fc4 = nn.Dense(500, x_dim)
        self.fill = P.Fill()
        self.exp = P.Exp()

    def ones(self, shape):
        return self.fill(mstype.float32, shape, 1.)

    def zeros(self, shape):
        return self.fill(mstype.float32, shape, 0.)

    def construct(self, observed):
        nodes = {}
        nodes = self.normal('z', mean=self.zeros((self.batch_size, self.z_dim)), sd=self.ones((self.batch_size, self.z_dim)), sample_shape=(()), observed=observed, nodes=nodes, reparameterize=False)
        z = nodes['z'][0]
        x_logits = self.act2(self.fc2(self.act1(self.fc1(z)))))
        x_mean = self.fc3(x_logits)
        x_sd = self.exp(self.fc4(x_logits))
        nodes = self.normal('x', mean=x_mean, sd=x_sd, sample_shape=(()), observed=observed, nodes=nodes, reparameterize=True)
        return nodes


class Variational(BayesianNet):
    def __init__(self, x_dim, z_dim, batch_size):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.fc1 = nn.Dense(x_dim, 500)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Dense(500, 500)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Dense(500, z_dim * 2)
        self.fill = P.Fill()
        self.exp = P.Exp()

    def ones(self, shape):
        return self.fill(mstype.float32, shape, 1.)

    def zeros(self, shape):
        return self.fill(mstype.float32, shape, 0.)

    def construct(self, observed):
        x = observed['x']
        nodes = {}
        z_logits = self.act2(self.fc2(self.act1(self.fc1(x))))
        z_mean = self.fc3(z_logits)
        z_sd = self.exp(self.fc4(z_logits))
        nodes = self.normal('z', mean=z_mean, sd=z_sd, sample_shape=(()), observed=observed, nodes=nodes, reparameterize=True)
        return nodes


class ELBO(nn.Cell):
    def __init__(self, generator, variational):
        super(ELBO, self).__init__()
        self.generator = generator
        self.variational = variational
        self.reshape_op = P.Reshape()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.square = P.Square()

    def log_joint(self, nodes):
        log_joint_ = 0.
        for _, logp in nodes:
            log_joint_ += logp
        return log_joint_

    def construct(self, x):
        x = self.reshape_op(x, (32, 32*32))
        nodes_q = self.variational({'x': x})
        z, logqz = nodes_v['z']
        nodes_p = self.generator({'x': x, 'z': z})
        logpxz = self.log_joint(nodes_p)
        elbo = self.reduce_mean(logpxz - logqz)
        return -elbo


def main():
    # We currently support pynative mode with device GPU 

    #context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU') 
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend') 
    epoch_size = 1
    batch_size = 32
    mnist_path = "/home/wds/iter-10/MNIST_Data"
    repeat_size = 1

    # Define model parameters
    z_dim = 40
    x_dim = 32*32

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    network = ELBO(generator, variational)

    # define loss
    # learning rate setting
    lr = 0.001
    net_loss = ReduceMeanLoss()

    # define the optimizer
    net_opt = nn.Adam(network.trainable_params(), lr)

    model = Model(network, net_loss, net_opt)

    ds_train = create_dataset(os.path.join(mnist_path, "train"), batch_size, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)

    # Bellow is the sample code for generating x.
    iterator = ds_train.create_tuple_iterator()
    for item in iterator:
        batch_x = item[0].reshape(32, 32*32)
        break
    nodes_q = network.variational({'x': Tensor(batch_x)})
    z, _ = nodes_q['z']
    nodes_p = self.generator({'z': z})
    sample = nodes_p['x'][0].asnumpy()
    if not os.path.exists('result'):
        os.mkdir('result')
    save_img(batch_x, 'result/origin_x.png')
    save_img(sample, 'result/reconstruct_x.png')

    for i in range(4):
        nodes_p = self.generator({})
        sample = nodes_p['x'][0].asnumpy()
        samples = sample if i == 0 else np.concatenate([samples, sample], axis=0)
    save_img(samples, 'result/sample_x.png', num=4*batch_size)

if __name__ == '__main__':
    main()

	


