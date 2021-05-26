import paddle
import paddle.fluid as fluid
import unittest
import numpy as np

class TestInvertibleTransform(unittest.TestCase):
    """
    Base Test for all Invertible Transforms
    """

    def assert_tensor_is_good(self, tensor, shape=None):
        self.assertIsInstance(tensor, paddle.Tensor)
        self.assertFalse(fluid.layers.has_nan(tensor))
        self.assertFalse(fluid.layers.has_inf(tensor))
        if shape is not None:
            self.assertEqual(tensor.shape, shape)

    def assert_invertible(self, *inputs, transform=None, decimal=7):
        z, log_det = transform.forward(*inputs, reverse=False)

        if not isinstance(z, tuple):
            xr, _ = transform.forward(z, reverse=True)
            self.assert_tensor_is_good(z)
            self.assert_tensor_is_good(xr)
            np.testing.assert_almost_equal(inputs[0].numpy(), xr.numpy(), decimal=decimal)
        else:
            xr, _ = transform.forward(*z, reverse=True)
            for i, _x in enumerate(z):
                self.assert_tensor_is_good(z[i])
                self.assert_tensor_is_good(xr[i])
                np.testing.assert_almost_equal(inputs[i].numpy(), xr[i].numpy(), decimal=decimal)