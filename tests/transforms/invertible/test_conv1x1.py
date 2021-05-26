import paddle
import paddle.nn as nn
import unittest
from tests.transforms.invertible.base import TestInvertibleTransform

from zhusuan.transforms.invertible import *

class TestConv1x1(TestInvertibleTransform):

    def test_invertible(self):
        batch_size = 10
        num_channel = 3
        num_features = 10

        x = paddle.randn(shape=[batch_size, num_channel, num_features])
        t = InvertibleConv1x1(num_channel)
        self.assert_invertible(x, transform=t, decimal=6)