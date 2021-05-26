import paddle
import paddle.nn as nn
import unittest
from tests.transforms.invertible.base import TestInvertibleTransform

from zhusuan.transforms.invertible import *


class TestBatchNorm(TestInvertibleTransform):

    def test_invertible(self):
        batch_size = 10
        in_out_dim = 10

        x = paddle.randn(shape=[batch_size, in_out_dim])
        t = BatchNorm(input_size=in_out_dim)
        self.assert_invertible(x, transform=t)


class TestActNorm(TestInvertibleTransform):

    def test_invertible(self):
        batch_size = 10
        num_feature = 10
        x = paddle.randn(shape=[batch_size, num_feature])
        t = ActNorm()
        self.assert_invertible(x, transform=t, decimal=6)
