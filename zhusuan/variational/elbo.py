""" ELBO """

import paddle
import paddle.fluid as fluid


class ELBO(paddle.nn.Layer):

    def __init__(self, generator, variational, estimator='sgvb'):
        super(ELBO, self).__init__()
        supported_estimator = ['sgvb', 'reinforce']

        self.generator = generator
        self.variational = variational

        if estimator not in supported_estimator:
            raise NotImplementedError()
        self.estimator = estimator
        if estimator == 'reinforce':
            mm = paddle.zeros(shape=[], dtype='float32')
            self.register_buffer('moving_mean', mm)

    def log_joint(self, nodes):
        log_joint_ = None
        for n_name in nodes.keys():
            try:
                log_joint_ += nodes[n_name].log_prob()
            except:
                log_joint_ = nodes[n_name].log_prob()

        return log_joint_

    def forward(self, observed, reduce_mean=True, baseline=None, variance_reduction=True, decay=0.8):
        nodes_q = self.variational(observed).nodes

        _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        nodes_p = self.generator(_observed).nodes

        logpxz = self.log_joint(nodes_p)
        logqz = self.log_joint(nodes_q)

        if self.estimator == "sgvb":
            return self.sgvb(logpxz, logqz, reduce_mean)
        elif self.estimator == "reinforce":
            return self.reinforce(logpxz, logqz, reduce_mean, baseline, variance_reduction, decay)

    def sgvb(self, logpxz, logqz, reduce_mean=True):
        if len(logqz.shape) > 0 and reduce_mean:
            elbo = fluid.layers.reduce_mean(logpxz - logqz)
        else:
            elbo = logpxz - logqz
        return -elbo

    def reinforce(self, logpxz, logqz, reduce_mean=True, baseline=None, variance_reduction=True, decay=0.8):
        l_signal = logpxz - logqz
        baseline_cost = None

        if variance_reduction:
            if baseline is not None:
                baseline_cost = 0.5 * paddle.square(
                    l_signal.detach() - baseline
                )
                if len(logqz.shape) > 0 and reduce_mean:
                    baseline_cost = fluid.layers.reduce_mean(baseline_cost)
            l_signal = l_signal - baseline

            # TODO: extend to non-scalar
            bc = fluid.layers.reduce_mean(l_signal)
            # Moving average
            self.moving_mean -= (1 - decay) * (self.moving_mean - bc)
            l_signal -= self.moving_mean

        cost = -logpxz - l_signal.detach() * logqz

        if len(logqz.shape) > 0 and reduce_mean:
            cost = fluid.layers.reduce_mean(cost)

        if baseline_cost is not None:
            return cost, baseline_cost
        else:
            return cost
