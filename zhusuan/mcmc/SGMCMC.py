import math
from collections import namedtuple

import paddle
import paddle.fluid as fluid

__all__ = [
    "SGMCMC",
]


class SGMCMC(paddle.nn.Layer):
    """
        Base Class for SGMCMC method
    """

    def __init__(self):
        super().__init__()
        self.t = 0

    def _update(self, bn, observed):
        raise NotImplementedError()

    def forward(self, bn, observed, resample=False, step=1):
        if resample:
            self.t = 0
            bn.forward(observed)
            self.t += 1

            self._latent = {k: v.tensor for k, v in bn.nodes.items() if k not in observed.keys()}
            self._latent_k = self._latent.keys()
            self._var_list = [self._latent[k] for k in self._latent_k]
            # self._var_list = [fluid.layers.zeros(self._latent[k].shape, dtype='float32')
            # for k in self._latent_k]
            sample_ = dict(zip(self._latent_k, self._var_list))

            for i in range(len(self._var_list)):
                self._var_list[i] = self._var_list[i].detach()
                self._var_list[i].stop_gradient = False

            return sample_

        for s in range(step):
            self._update(bn, observed)
            self.t += 1

        sample_ = dict(zip(self._latent_k, self._var_list))
        return sample_

    def initialize(self):
        self.t = 0

    def sample(self, bn, observed, resample=False, step=1):
        """
        Return the sampling `Operation` that runs a SGMCMC iteration and the
        statistics collected during it, given the log joint function (or a
        :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance), observed
        values and latent variables.
        """
        return self.forward(bn, observed, resample, step)
