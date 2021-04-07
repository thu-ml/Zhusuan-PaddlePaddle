import math

import paddle
import paddle.fluid as fluid

from zhusuan.mcmc.SGMCMC import SGMCMC

__all__ = [
    "SGHMC",
]

class SGHMC(SGMCMC):
    def __init__(self, learning_rate, friction=0.25, variance_estimate=0.,
                 n_iter_resample_v=20, second_order=True):
        super(SGHMC, self).__init__()
        # self.lr = paddle.to_tensor([learning_rate], dtype='float32')
        # self.lr_num = learning_rate
        self.lr = learning_rate
        self.alpha = friction
        self.beta = variance_estimate
        # self.alpha = paddle.to_tensor([friction], dtype='float32')
        # self.beta = paddle.to_tensor([variance_estimate], dtype='float32')
        if n_iter_resample_v is None:
            n_iter_resample_v = 0
        self.n_iter_resample_v = n_iter_resample_v
        self.second_order = second_order
        self.vs = None

    def _update(self, bn, observed):
        if not self.vs:
            self.vs = [paddle.normal(shape=paddle.shape(q), mean=0., std=math.sqrt(self.lr)) for q in self._var_list]

        gaussian_term = None
        for i, _ in enumerate(self.vs):
            if self.n_iter_resample_v != 0:
                if self.t % self.n_iter_resample_v == 0:
                    self.vs[i] = paddle.normal(shape=paddle.shape(self.vs[i]), mean=0., std=math.sqrt(self.lr))
            gaussian_term = paddle.normal(shape=paddle.shape(self.vs[i]),
                                          mean=0.,
                                          std=math.sqrt(2 * (self.alpha - self.beta) * self.lr))
            if self.second_order:
                self._var_list[i] = self._var_list[i] + 0.5 * self.vs[i]

        observed_ = {**dict(zip(self._latent_k, self._var_list)), **observed}
        bn.forward(observed_)
        log_joint_ = bn.log_joint()
        grad = paddle.grad(log_joint_, self._var_list)

        for i, _ in enumerate(grad):
            if not self.second_order:
                self.vs[i] = (1 - self.alpha) * self.vs[i] + self.lr * grad[i] + gaussian_term
                self._var_list[i] = self._var_list[i] + self.vs[i]
            else:
                decay_half = math.exp(-0.5 * self.alpha)
                self.vs[i] = decay_half * (decay_half * self.vs[i] + self.lr * grad[i] + gaussian_term)
                self._var_list[i] = self._var_list[i] + 0.5 * self.vs[i]