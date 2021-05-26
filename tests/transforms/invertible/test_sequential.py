import paddle
import paddle.nn as nn
import unittest
from tests.transforms.invertible.base import TestInvertibleTransform

from zhusuan.transforms.invertible.base import InvertibleTransform
from zhusuan.transforms.invertible.sequential import Sequential


class TestSequential(TestInvertibleTransform):
    def test_invertible(self):
        class SimpleTransform(InvertibleTransform):
            def __init__(self):
                super(SimpleTransform, self).__init__()

            def _forward(self, x, v, **kwargs):
                return 2 * x + 1, v + 1, None

            def _inverse(self, y, v, **kwargs):
                return (y - 1) / 2, v - 1, None

        class SimpleTransform2(InvertibleTransform):
            def __init__(self):
                super(SimpleTransform2, self).__init__()

            def _forward(self, x, v, **kwargs):
                return (2 * x + 1, v + 1), None

            def _inverse(self, y, v, **kwargs):
                return ((y - 1) / 2, v - 1), None

        layers = []
        # Two different way to pass vars
        for i in range(5):
            layers.append(SimpleTransform())
        for i in range(5):
            layers.append(SimpleTransform2())
        transform = Sequential(layers)
        x = paddle.randn(shape=[10, 10])
        v = paddle.randn(shape=[10, 10])
        self.assert_invertible(x, v, transform=transform, decimal=5)
