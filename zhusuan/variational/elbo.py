""" ELBO """

import paddle
import paddle.fluid as fluid


class ELBO(paddle.nn.Layer):

    def __init__(self, generator, variational, estimator='sgvb', transform=None, transform_var=[], auxillary_var=[]):
        super(ELBO, self).__init__()
        supported_estimator = ['sgvb', 'reinforce']

        self.generator = generator
        self.variational = variational

        if estimator not in supported_estimator:
            raise NotImplementedError()
        self.estimator = estimator
        if estimator == 'reinforce':
            mm = paddle.zeros(shape=[1], dtype='float32')
            ls = paddle.zeros(shape=[1], dtype='int32')
            self.register_buffer('moving_mean', mm)
            self.register_buffer('local_step', ls)
            self.moving_mean.stop_gradient = True

        if transform:
            self.transform = transform
            self.transform_var = transform_var
            self.auxillary_var = auxillary_var
            # assert isinstance(self.transform, dict)
        else:
            self.transform = None

    def log_joint(self, nodes):
        log_joint_ = None
        for n_name in nodes.keys():
            try:
                log_joint_ += nodes[n_name].log_prob()
            except:
                log_joint_ = nodes[n_name].log_prob()
        return log_joint_

    def forward(self, observed, reduce_mean=True, **kwargs):
        nodes_q = self.variational(observed).nodes
        log_det = None
        if self.transform is not None:
            _transformed_inputs = {}
            _v_inputs = {}

            # Build input tuple for flow
            flow_inputs = []
            for k in self.transform_var:
                # Only latent variable can be transformed
                assert k not in observed.keys()
                assert k in nodes_q.keys()
                flow_inputs.append(nodes_q[k].tensor)
            for k in self.auxillary_var:
                flow_inputs.append(self.variational.cache[k])
            flow_inputs = tuple(flow_inputs)

            # Transform
            output, log_det = self.transform(*flow_inputs, reverse=False)
            assert len(output) == len(self.transform_var)  # All transformed var should be returned
            for k in self.transform_var:
                _transformed_inputs[k] = output[k]

            for k, v in nodes_q.items():
                if k not in _transformed_inputs.keys():
                    _v_inputs[k] = v.tensor
            _observed = {**_transformed_inputs, **_v_inputs, **observed}
            nodes_p = self.generator(_observed).nodes
            logpxz = self.log_joint(nodes_p)
            logqz = self.log_joint(nodes_q)

        else:
            _v_inputs = {k: v.tensor for k, v in nodes_q.items()}
            _observed = {**_v_inputs, **observed}
            nodes_p = self.generator(_observed).nodes
            logpxz = self.log_joint(nodes_p)
            logqz = self.log_joint(nodes_q)

        if self.estimator == "sgvb":
            return self.sgvb(logpxz, logqz, reduce_mean, log_det)
        elif self.estimator == "reinforce":
            return self.reinforce(logpxz, logqz, reduce_mean, **kwargs)

    def sgvb(self, logpxz, logqz, reduce_mean=True, log_det=None):
        if len(logqz.shape) > 0 and reduce_mean:
            elbo = fluid.layers.reduce_mean(logpxz - logqz)
        else:
            elbo = logpxz - logqz

        if log_det is not None:
            elbo += paddle.mean(paddle.sum(log_det, axis=1)).squeeze()

        return -elbo

    def reinforce(self, logpxz, logqz, reduce_mean=True, baseline=None, variance_reduction=True, decay=0.8):
        decay_tensor = paddle.ones(shape=[1], dtype='float32') * decay
        l_signal = logpxz - logqz
        l_signal.stop_gradient = True
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
            if len(logqz.shape) > 0 and reduce_mean:
                bc = fluid.layers.reduce_mean(l_signal)
            else:
                bc = l_signal
            # Moving average
            self.moving_mean -= (self.moving_mean - bc) * (1.0 - decay)
            self.local_step += 1
            bias_factor = 1 - paddle.pow(decay_tensor, self.local_step)
            self.moving_mean /= bias_factor
            l_signal -= self.moving_mean.detach()
        l_signal = l_signal.detach()
        l_signal.stop_gradient = True
        cost = -logpxz - l_signal * logqz
        if baseline_cost is not None:
            if len(logqz.shape) > 0 and reduce_mean:
                loss = fluid.layers.reduce_mean(cost + baseline_cost)
            else:
                loss = cost + baseline_cost
            return loss, fluid.layers.reduce_mean(logpxz - logqz)
        else:
            if len(logqz.shape) > 0 and reduce_mean:
                cost = fluid.layers.reduce_mean(cost)
            return cost
