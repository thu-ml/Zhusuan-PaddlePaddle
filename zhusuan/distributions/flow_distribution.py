import paddle
import paddle.fluid as fluid
import numpy as np

from .base import Distribution

__all__ = [
    'FlowDistribution',
]


class FlowDistribution(Distribution):
    def __init__(self, latents=None, transformation=None, flow_kwargs=None, dtype='float32', group_ndims=0, **kwargs):
        self._latents = latents
        self._transformation = transformation
        self._flow_kwargs = flow_kwargs

        super(FlowDistribution, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    def _sample(self, n_samples=-1, **kwargs):
        if 'shape' in kwargs.keys():
            _shape = kwargs['shape']
            z = self._latents.sample(shape=_shape)
            x, _ = self._transformation.forward(z, reverse=True, **kwargs)
            return x[0]
        elif n_samples != -1:
            z = self._latents.sample(n_samples)
            x, _ = self._transformation.forward(z, reverse=True, **kwargs)
            return x[0]
        else:
            return 0.

    def _log_prob(self, *given, **kwargs):
        z, log_det_J = self._transformation.forward(*given, **kwargs, reverse=False)
        log_ll = paddle.sum(self._latents.log_prob(z[0]) + log_det_J, axis=1)
        return log_ll
