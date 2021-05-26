# from __future__ import absolute_import
# from __future__ import division
#
# import paddle
# import paddle.fluid as fluid
# import paddle.nn.functional as F
# import numpy as np
# import scipy
#
# from zhusuan.transforms.invertible.base import InvertibleTransform
#
#
# class Sequential(InvertibleTransform):
#     def __init__(self, layers):
#         super().__init__()
#         self.layers = layers
#
#     def _forward(self, x, **kwargs):
#         batch_size = int((x[0] if isinstance(x, tuple) else x).shape[0])
#         logd_terms = []
#         for f in self.layers:
#             assert isinstance(f, InvertibleTransform)
#             x, l = f.forward(x, **kwargs, reverse=False)
#             if l is not None:
#                 assert l.shape == [batch_size]
#                 logd_terms.append(l)
#         return x, sum(logd_terms) if logd_terms else paddle.zeros([])
#
#     def _inverse(self, y, **kwargs):
#         batch_size = int((y[0] if isinstance(y, tuple) else y).shape[0])
#         logd_terms = []
#         for f in self.layers:
#             assert isinstance(f, InvertibleTransform)
#             y, l = f.forward(y, **kwargs, reverse=True)
#             if l is not None:
#                 assert l.shape == [batch_size]
#                 logd_terms.append(l)
#         return y, sum(logd_terms) if logd_terms else paddle.zeros([])
#
#
# class BaseNorm(InvertibleTransform):
#     """
#         Base class for ActNorm (Glow) and PixNorm (Flow++).
#         The mean and inv_std get initialized using the mean and variance of the
#         first mini-batch. After the init, mean and inv_std are trainable parameters.
#         Adapted from:
#             > https://github.com/chrischute/flowplusplus
#     """
#
#     def __init__(self, num_channels, height, width):
#         super().__init__()
#         num_channels *= 2
#
#         self.register_buffer('is_inited', paddle.zeros(shape=[1]))
#         self.mean = self.create_parameter([1, num_channels, height, width])
#         self.inv_std = self.create_parameter([1, num_channels, height, width])
#         self.add_parameter("mean", self.mean)
#         self.add_parameter("inv_std", self.inv_std)
#         self.eps = 1e-6
#
#     def init_parameters(self, x):
#         if not self.training:
#             return
#         with paddle.no_grad():
#             mean, inv_std = self._get_moments(x)
#             self.mean = paddle.assign(mean)
#             self.inv_std = paddle.assign(inv_std)
#             self.is_inited += 1.
#
#     def _center(self, x, reverse=False):
#         return x + self.mean if reverse else x - self.mean
#
#     def _get_moments(self, x):
#         raise NotImplementedError()
#
#     def _scale(self, x, l, reverse=False):
#         raise NotImplementedError()
#
#     def _forward(self, x, **kwargs):
#         if isinstance(x, tuple):
#             x = paddle.concat(x, axis=1)
#         if not self.is_inited:
#             self.init_parameters(x)
#         l = kwargs['ldj']
#         assert l is not None
#         x, l = self._scale(x, l, reverse=False)
#         x = self._center(x, reverse=False)
#         x = paddle.chunk(x, 2, axis=1)
#         return x, l
#
#     def _inverse(self, y, **kwargs):
#         if isinstance(y, tuple):
#             y = paddle.concat(y, axis=-1)
#         if not self.is_inited:
#             self.init_parameters(y)
#         l = kwargs['ldj']
#         assert l is not None
#         y, l = self._scale(y, l, reverse=True)
#         y = self._center(y, reverse=True)
#         y = paddle.chunk(y, 2, axis=1)
#         return y, l
#
#
# class ActNorm(BaseNorm):
#     def __init__(self, num_channels):
#         super(ActNorm, self).__init__(num_channels, 1, 1)
#
#     def _get_moments(self, x):
#         mean = fluid.layers.reduce_mean(x, dim=[0, 2, 3], keep_dim=True)
#         var = fluid.layers.reduce_mean((x - mean) ** 2, dim=[0, 2, 3], keep_dim=True)
#         inv_std = 1. / (paddle.sqrt(var) + self.eps)
#         return mean, inv_std
#
#     def _scale(self, x, l, reverse=False):
#         if reverse:
#             x = x / self.inv_std
#             l = l - fluid.layers.reduce_sum(paddle.log(self.inv_std)) * x.shape[2] * x.shape[3]
#         else:
#             x = x * self.inv_std
#             l = l + fluid.layers.reduce_sum(paddle.log(self.inv_std)) * x.shape[2] * x.shape[3]
#         return x, l
#
# class InvConv2d(InvertibleTransform):
#     """
#         Invertible 1x1 Conv for 2D inputs. Originally described in Glow
#         (https://arxiv.org/abs/1807.03039). Because Paddle does not support
#         determinant, so only LU-decomposed version can be used
#     """
#     def __init__(self, num_channels):
#         super().__init__()
#         weight = np.random.randn(num_channels, num_channels)
#         q, _ = scipy.linalg.qr(weight)
#         w_p, w_l, w_u = scipy.linalg.lu(q.astype(np.float32))
#         w_s = np.diag(w_u)
#         w_u = np.triu(w_u, 1)
#         u_mask = np.triu(np.ones_like(w_u), 1)
#         l_mask = u_mask.T
#
#         w_p = paddle.to_tensor(w_p)
#         w_l = paddle.to_tensor(w_l)
#         w_s = paddle.to_tensor(w_s)
#         w_u = paddle.to_tensor(w_u)
#
#         self.register_buffer("w_p", w_p)
#         self.register_buffer("u_mask", paddle.to_tensor(u_mask))
#         self.register_buffer("l_mask", paddle.to_tensor(l_mask))
#         self.register_buffer("s_sign", paddle.sign(w_s))
#         self.register_buffer("l_eye", paddle.eye(l_mask.shape[0]))
#
#         self.weight_L = self.create_parameter(shape=w_l.shape,
#                                               default_initializer=paddle.nn.initializer.Assign(w_l))
#         self.weight_S = self.create_parameter(shape=w_s.shape,
#                                               default_initializer=paddle.nn.initializer.Assign(
#                                                                   paddle.log(paddle.abs(w_s))))
#         self.weight_U = self.create_parameter(shape=w_u.shape,
#                                               default_initializer=paddle.nn.initializer.Assign(w_u))
#         self.add_parameter("weight_L", self.weight_L)
#         self.add_parameter("weight_S", self.weight_S)
#         self.add_parameter("weight_U", self.weight_U)
#
#     def calc_weight(self):
#         weight = (
#             self.w_p
#             @ (self.weight_L * self.l_mask + self.l_eye)
#             @ ((self.weight_U * self.u_mask) + paddle.diag(self.s_sign * paddle.exp(self.w_s)))
#         )
#         return weight.unsqueeze(axis=2).unsqueeze(axis=3)
#
#     def _forward(self, x, **kwargs):
#         _, _, height, width = x.shape
#         weight = self.calc_weight()
#         x = F.conv2d(x, weight)
#         logdet = height * width * paddle.sum(self.weight_S)
#         return x, logdet
#
#     def _inverse(self, y, **kwargs):
#         weight = self.calc_weight()
#         return F.conv2d(y, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
#
# class ZeroConv2d(paddle.nn.Layer):
#     def __init__(self, in_channel, out_channel, padding=1):
#         super().__init__()
#
#         self.conv = paddle.nn.Conv2D(in_channel, out_channel, 3, padding=0)
#
#
#
# class AffineCoupling(InvertibleTransform):
#     pass
#
#
