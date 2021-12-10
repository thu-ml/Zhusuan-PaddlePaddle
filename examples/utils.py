import os
import math
import gzip

import paddle
import progressbar
import six
import numpy as np
from six.moves import urllib, range
from six.moves import cPickle as pickle
from PIL import Image
from matplotlib import pyplot as plt

from paddle.io import Dataset, TensorDataset, DataLoader

pbar = None
examples_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(examples_dir, "data")
data_path = os.path.join(data_dir, "mnist.pkl.gz")


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def standardize(data_train, data_test):
    """
    Standardize a datasets to have zero mean and unit standard deviation.
    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.
    :return: (train_set, test_set, mean, std), The standardized datasets and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    data_test_standardized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_standardized, data_test_standardized, mean, std


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
    # urllib.request.urlretrieve(url, path, show_progress)
    urllib.request.urlretrieve(url, path)


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


def logit(x, alpha):
    x_ = alpha + (1 - 2 * alpha) * x
    return np.log(x_ / (1 - x_))


def load_mnist_realval(path=data_path, one_hot=True, dequantify=False, logit_transform=False):
    """
    Loads the real valued MNIST datasets.

    :param path: Path to the datasets file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).

    :return: The MNIST datasets.
    """
    alpha = 1.0e-6
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
    # x_train, t_train = train_set[0][:64*500], train_set[1][:64*500]
    # x_valid, t_valid = valid_set[0][:64*50], valid_set[1][:64*50]
    # x_test, t_test = test_set[0][:64*10], test_set[1][:64*10]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    if logit_transform:
        x_train = logit(x_train, alpha)
        x_valid = logit(x_valid, alpha)
        x_test = logit(x_test, alpha)

    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train, t_transform(t_train), x_valid, t_transform(t_valid), \
           x_test, t_transform(t_test)


def save_img(data, name):
    """
    Visualize data and save to target files
    Args:
        data: nparray of size (num, size, size)
        name: ouput file name
        size: image size
        num: number of images
    """

    size = int(data.shape[1] ** .5)
    num = data.shape[0]
    col = int(num / 8)
    row = 8

    imgs = Image.new('L', (size * col, size * row))
    for i in range(num):
        j = i / 8
        img_data = data[i]
        img_data = np.resize(img_data, (size, size))
        img_data = img_data * 255
        img_data = img_data.astype(np.uint8)
        im = Image.fromarray(img_data, 'L')
        imgs.paste(im, (int(j) * size, (i % 8) * size))
    imgs.save(name)


class MNISTDataset(Dataset):
    def __init__(self, mode='train'):
        super(MNISTDataset, self).__init__()
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


def load_uci_boston_housing(path, dtype=np.float32):
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://archive.ics.uci.edu/ml/'
                         'machine-learning-databases/housing/housing.data',
                         path)

    data = np.loadtxt(path)
    data = data.astype(dtype)
    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]
    x_test, y_test = data[index_test, :-1], data[index_test, -1]

    return x_train, y_train, x_val, y_val, x_test, y_test


def fetch_dataloaders(dataset_name, batch_size, dequantify=True, logit_transform=True):
    if dataset_name in ['MNIST']:
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist_realval(dequantify=dequantify, logit_transform=logit_transform)
        lam = 1.0e-6

        # join train and val data again
        x_train = np.concatenate((x_train, x_valid), axis=0).astype(np.float32)
        y_train = np.concatenate((y_train, y_valid), axis=0).astype(np.float32)

        # construct datasets
        train_dataset = TensorDataset([paddle.to_tensor(x_train), paddle.to_tensor(y_train)])
        test_dataset = TensorDataset([paddle.to_tensor(x_test), paddle.to_tensor(y_test)])

        n_dims = (1, 28, 28)
        label_size = 10

    else:
        raise ValueError('Unrecognized datasets')

    train_dataset.input_dims = n_dims
    train_dataset.input_size = int(np.prod(n_dims))
    train_dataset.label_size = label_size
    train_dataset.lam = lam

    test_dataset.input_dims = n_dims
    test_dataset.input_size = int(np.prod(n_dims))
    test_dataset.label_size = label_size
    test_dataset.lam = lam

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


@paddle.no_grad()
def save_image(tensor, filename, nrow=8, padding=2, pad_value=0):
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = math.ceil(float(nmaps) / xmaps)
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = Image.new('L', (width * xmaps, height * ymaps))
    k = 0
    tensor = (tensor * 255 + 0.5).clip(0, 255).transpose(perm=[0, 2, 3, 1])
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            data = tensor[k].numpy()
            data = np.resize(data, (tensor.shape[1], tensor.shape[2]))
            data = data.astype(np.uint8)
            im = Image.fromarray(data, 'L')
            grid.paste(im, (width * x, height * y))
            k = k + 1
    grid.save(filename)