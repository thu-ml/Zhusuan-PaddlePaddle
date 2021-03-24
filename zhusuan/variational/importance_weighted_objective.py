""" ELBO """

import paddle
import paddle.fluid as fluid

from zhusuan import log_mean_exp

__all__ = [
    'iw_objective',
    'ImportanceWeightedObjective',
]


class ImportanceWeightedObjective(paddle.nn.Layer):
    def __init__(self, generator, variational, axis=None, estimator='sgvb'):
        super().__init__()
        supported_estimator = ['sgvb', 'vimco']

        self.generator = generator
        self.variational = variational

        if axis is None:
            raise ValueError(
                "ImportanceWeightedObjective is a multi-sample objective, "
                "the `axis` argument must be specified.")
        self._axis = axis

        if estimator not in supported_estimator:
            raise NotImplementedError()
        self.estimator = estimator

    def log_joint(self, nodes):
        log_joint_ = None
        for n_name in nodes.keys():
            try:
                log_joint_ += nodes[n_name].log_prob()
            except:
                log_joint_ = nodes[n_name].log_prob()

        return log_joint_

    def forward(self, observed, reduce_mean=True):
        nodes_q = self.variational(observed).nodes

        _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        nodes_p = self.generator(_observed).nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)

        if self.estimator == 'sgvb':
            return self.sgvb(logpxz, logqz, reduce_mean)
        else:
            return self.vimco(logpxz, logqz, reduce_mean)

    def sgvb(self, logpxz, logqz, reduce_mean=True):
        lower_bound = logpxz - logqz

        if self._axis is not None:
            lower_bound = log_mean_exp(lower_bound, self._axis)

        if reduce_mean:
            return fluid.layers.reduce_mean(-lower_bound)
        else:
            return -lower_bound

    def vimco(self, logpxz, logqz, reduce_mean=True):
        log_w = logpxz - logqz
        l_signal = log_w

        # check size along the sample axis
        err_msg = "VIMCO is a multi-sample gradient estimator, size along " \
                  "`axis` in the objective should be larger than 1."
        try:
            _shape = paddle.shape(l_signal)
            _ = _shape[self._axis:self._axis + 1]
            if _shape[self._axis] < 2:
                raise ValueError(err_msg)
        except:
            raise ValueError(err_msg)

        # compute variance reduction term
        mean_expect_signal = (fluid.layers.reduce_sum(l_signal, self._axis, keep_dim=True) - l_signal) \
                            / paddle.cast(paddle.shape(l_signal)[self._axis] - 1, l_signal.dtype)
        x, sub_x = l_signal, mean_expect_signal
        n_dim = paddle.cast(paddle.rank(x), dtype='int32')
        n_dim = paddle.unsqueeze(n_dim, [-1])
        axis_dim_mask = paddle.cast(fluid.one_hot(paddle.to_tensor(self._axis, dtype='int32'), n_dim), 'bool')
        original_mask = paddle.cast(fluid.one_hot(n_dim - 1, n_dim), 'bool')
        axis_dim = paddle.ones(n_dim, 'int32') * self._axis
        originals = paddle.ones(n_dim, 'int32') * (n_dim - 1)
        perm = paddle.where(paddle.squeeze(original_mask, axis=[-1]), axis_dim, paddle.arange(n_dim, dtype='int32'))
        perm = paddle.where(paddle.squeeze(axis_dim_mask, axis=[-1]), originals, perm)
        multiples = paddle.concat([paddle.ones(n_dim, 'int32'), paddle.shape(x)[self._axis]], 0)

        x = paddle.transpose(x, perm.numpy().tolist())
        sub_x = paddle.transpose(sub_x, perm.numpy().tolist())
        x_ex = paddle.tile(paddle.unsqueeze(x, n_dim), multiples)
        x_ex = x_ex - paddle.diag(x) + paddle.diag(sub_x)
        control_variate = paddle.transpose(log_mean_exp(x_ex, [n_dim - 1]), perm.numpy().tolist())

        # variance reduced objective
        l_signal = log_mean_exp(l_signal, self._axis, keepdims=True) - control_variate
        fake_term = fluid.layers.reduce_sum(logqz * l_signal.detach(), self._axis)
        cost = -fake_term - log_mean_exp(log_w, self._axis)

        return cost


iw_objective = 'ImportanceWeightedObjective',
