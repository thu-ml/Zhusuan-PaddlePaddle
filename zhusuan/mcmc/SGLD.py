import math

import paddle
import paddle.fluid as fluid

from zhusuan.mcmc.SGMCMC import SGMCMC

__all__ = [
    "SGLD",
    "PSGLD",
]


class SGLD(SGMCMC):
    """
        SGLD
    """

    def __init__(self, learning_rate):
        super().__init__()
        self.lr = paddle.to_tensor(learning_rate, dtype='float32')
        self.lr_min = paddle.to_tensor(1e-4, dtype='float32')


    def _update(self, bn, observed):
        observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
        bn.forward(observed_)

        log_joint_ = bn.log_joint()
        grad = paddle.grad(log_joint_, self._var_list)

        for i, _ in enumerate(grad):
            # _lr = max(self.lr_min, self.lr / math.sqrt(self.t))
            # _lr = self.lr / math.sqrt(self.t)
            epsilon = paddle.normal(shape=paddle.shape(self._var_list[i]), mean=0.0, std=math.sqrt(self.lr))
            self._var_list[i] = self._var_list[i] + 0.5 * self.lr * grad[i] + epsilon
            self._var_list[i] = self._var_list[i].detach()
            self._var_list[i].stop_gradient = False


class PSGLD(SGLD):
    """
        PSGLD with RMSprop preconditioner
    """

    def __init__(self, learning_rate, decay=0.9, epsilon=1e-3):
        super().__init__(learning_rate)
        self.aux = None
        self.decay = decay
        self.epsilon = epsilon

    def _update(self, bn, observed):
        if not self.aux:
            self.aux = [paddle.zeros_like(q) for q in self._var_list]
        observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
        bn.forward(observed_)

        log_joint_ = bn.log_joint()
        grad = paddle.grad(log_joint_, self._var_list)

        for i, _ in enumerate(grad):
            self.aux[i] = self.decay * self.aux[i] + (1 - self.decay) * paddle.pow(grad[i], 2)
            g = 1 / (self.epsilon + paddle.sqrt(self.aux[i]))
            e = paddle.normal(shape=paddle.shape(self._var_list[i]), mean=0.0, std=paddle.sqrt(self.lr * g))
            self._var_list[i] = self._var_list[i] + 0.5 * self.lr * g * grad[i] + e
            self._var_list[i] = self._var_list[i].detach()
            self._var_list[i].stop_gradient = False
