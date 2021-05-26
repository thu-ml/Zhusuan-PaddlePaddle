import paddle
import paddle.nn as nn
import unittest
from tests.transforms.invertible.base import TestInvertibleTransform

from zhusuan.transforms.invertible import *


class TestMADE(TestInvertibleTransform):

    def test_invertible(self):
        batch_size = 10
        in_out_dim = 10
        hidden_size = 20
        n_hidden = 3

        x = paddle.randn(shape=[batch_size, in_out_dim])
        t = MADE(in_out_dim, hidden_size, n_hidden)
        self.assert_invertible(x, transform=t, decimal=5)