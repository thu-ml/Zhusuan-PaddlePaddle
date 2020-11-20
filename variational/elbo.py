""" ELBO """

import paddle
import paddle.fluid as fluid

class ELBO(paddle.nn.Layer):
    def __init__(self, generator, variational):
        super(ELBO, self).__init__()
        self.generator = generator
        self.variational = variational

    def log_joint(self, nodes, reduce_mean=False):
        """
            reduce_mean: if set to True, reduce the log_joint_ for acceleration
        """
        if not reduce_mean:
            log_joint_ = None
            for n_name in nodes.keys():
                try:
                    log_joint_ += nodes[n_name][1]
                except:
                    log_joint_ = nodes[n_name][1]
        else:
            log_joint_ = 0.
            for n_name in nodes.keys():
                log_joint_ += fluid.layers.reduce_mean(nodes[n_name][1], dim=-1)

        return log_joint_

    def forward(self, observed):
        #print(observed)
        nodes_q = self.variational(observed).nodes
        #print(nodes_q)

        _v_inputs = {k:v[0] for k,v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        nodes_p = self.generator(_observed).nodes
        #print('nodes_p.keys: ', nodes_p.keys())

        logpxz = self.log_joint(nodes_p, reduce_mean=True)
        logqz = self.log_joint(nodes_q, reduce_mean=True)
        elbo = fluid.layers.reduce_mean(logpxz - logqz)

        return -elbo

