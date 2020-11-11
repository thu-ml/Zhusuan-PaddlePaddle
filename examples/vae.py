
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

from framework.bn import BayesianNet
from variational.elbo import ELBO


device = paddle.set_device('gpu') # or 'gpu'
paddle.disable_static(device)

pbar = None
examples_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(examples_dir, "data")

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        if total_size > 0:
            prefixes = ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi')
            power = min(int(math.log(total_size, 2) / 10), len(prefixes) - 1)
            scaled = float(total_size) / (2 ** (10 * power))
            total_size_str = '{:.1f} {}B'.format(scaled, prefixes[power])
            try:
                marker = '█'
            except UnicodeEncodeError:
                marker = '*'
            widgets = [
                progressbar.Percentage(),
                ' ', progressbar.DataSize(),
                ' / ', total_size_str,
                ' ', progressbar.Bar(marker=marker),
                ' ', progressbar.ETA(),
                ' ', progressbar.AdaptiveTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets,
                                           max_value=total_size)
        else:
            widgets = [
                progressbar.DataSize(),
                ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                ' ', progressbar.Timer(),
                ' ', progressbar.AdaptiveTransferSpeed(),
            ]
            pbar = progressbar.ProgressBar(widgets=widgets,
                                           max_value=progressbar.UnknownLength)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path, show_progress)

def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.

    :param x: 1-D Numpy array of type int.
    :param depth: A int.

    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret

def load_mnist_realval(path, one_hot=True, dequantify=False):
    """
    Loads the real valued MNIST dataset.

    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).

    :return: The MNIST dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
        x_test, t_transform(t_test)



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

        self.fc3 = paddle.nn.Linear(500, x_dim)
        # self.act3 = paddle.nn.Sigmoid()
        self.fc4 = paddle.nn.Linear(500, x_dim)
        # self.fc5 = paddle.nn.Linear(500, x_dim)
        # self.fc4 = paddle.nn.Linear(500, x_dim)
        # self.fc5 = paddle.nn.Linear(500, x_dim)

        self.sd = paddle.to_variable(fluid.layers.ones(shape=(self.batch_size, self.z_dim), dtype='float32'))
        self.mean = paddle.to_variable(fluid.layers.zeros(shape=(self.batch_size, self.z_dim), dtype='float32'))

    def forward(self, observed):
        # Check mark, mean, std, shape
        sd = self.sd
        mean = self.mean
        if 'z' in observed.keys():
            sd  = paddle.to_variable(fluid.layers.ones(shape=(observed['z'].shape[0], self.z_dim), dtype='float32'))
            mean = paddle.to_variable(fluid.layers.zeros(shape=(observed['z'].shape[0], self.z_dim), dtype='float32'))

        nodes = {}
        nodes = self.Normal('z', mean=mean,
                            std=sd,
                            shape=(()), observation=observed,
                            nodes=nodes, reparameterize=False)
        z = nodes['z'][0]
        x_logits = self.act2(self.fc2(self.act1(self.fc1(z))))
        # print(x_logits.shape)
        x_mean = self.fc3(x_logits)
        # x_sd = self.fc5(x_logits)
        x_sd = paddle.exp(self.fc4(x_logits))
        nodes =self.Normal('x', mean=x_mean, std=x_sd, shape=(()), observation=observed, nodes=nodes, reparameterize = True)
        # nodes =self.Bernoulli('x', shape=(()), observation=observed, nodes=nodes, probs=x_logits)

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
        # z_sd = self.fc4(z_logits)
        z_sd = paddle.exp(self.fc4(z_logits))
        nodes = self.Normal('z', mean=z_mean, std=z_sd, shape=(()), observation=observed, nodes=nodes,
                            reparameterize=True)
        return nodes


class ReduceMeanLoss(paddle.nn.Layer):
    def __init__(self):
        super(ReduceMeanLoss, self).__init__()

    def forward(self, input, label):
        # label_cast = fluid.layers.cast(label, dtype = 'float32')
        # input_cast = fluid.layers.cast(input, dtype='float32')
        # # print([input_cast, label_cast])
        # loss = F.l1_loss(input_cast, label_cast,  reduction='mean')
        return input #loss #paddle.mean(loss)


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


def save_img(data, name):
    """
    Visualize data and save to target files
    Args:
        data: nparray of size (num, size, size)
        name: ouput file name
        size: image size
        num: number of images
    """

    size = int(data.shape[1]**.5)
    num = data.shape[0]
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




def main():

    epoch_size = 1
    batch_size = 32

    # Define model parameters
    z_dim = 500
    x_dim = 28*28

    # create the network
    generator = Generator(x_dim, z_dim, batch_size)
    variational = Variational(x_dim, z_dim, batch_size)
    network = ELBO(generator, variational)

    # define loss
    # learning rate setting
    lr = 0.0001
    net_loss = ReduceMeanLoss()

    # define the optimizer
    net_opt = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=network.parameters())

    # Model training
    model = paddle.Model(network)
    model.prepare(net_opt, net_loss)

    model.fit( train_data =MyDataset(mode='train'), #train_dataset, #MyDataset(mode='train'),
               eval_data = MyDataset(mode='valid'), #test_dataset,  #MyDataset(mode='valid'),
               batch_size = batch_size,
               epochs = epoch_size,
               eval_freq = 5,
               save_dir = '../model/',
               save_freq = 1,
               verbose = 2,
               shuffle = True,
               num_workers = 0,
               callbacks = None)

    # Model test
    test_data = MyDataset(mode='test')
    batch_x = test_data.data[0:72]

    # print(len(model.parameters()))
    # params_info = model.summary()

    generator, variational = list(model.network.children())
    # print([len(generator.parameters()), len(variational.parameters())])
    nodes_q = variational({'x': paddle.to_tensor(batch_x)})
    z, _ = nodes_q['z']
    nodes_p = generator({'z': z})

    sample = nodes_p['x'][0].numpy()
    if not os.path.exists('result'):
        os.mkdir('result')
    print([sample.shape, batch_x.shape])

    # plt.figure(0)
    # cv2.imshow('input',batch_x[0])
    print([np.max(batch_x[0]),np.min(batch_x[0])])
    print([np.max(sample[0]),np.min(sample[0])])

    result_fold = 'result'
    if not os.path.isdir(result_fold):
        os.mkdir(result_fold)
    save_img(batch_x, os.path.join(result_fold, 'origin_x.png' ))
    save_img(sample,  os.path.join(result_fold, 'reconstruct_x.png' ))


if __name__ == '__main__':
    main()