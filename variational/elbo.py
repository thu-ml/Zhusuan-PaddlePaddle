

""" ELBO """

import paddle
import paddle.fluid as fluid

class ELBO(paddle.nn.Layer):
    def __init__(self, generator, variational):
        super(ELBO, self).__init__()
        self.generator = generator
        self.variational = variational

    def log_joint(self, nodes):
        log_joint_ = 0.
        for n_name in nodes.keys():
            for node in nodes[n_name]:
                log_joint_ += fluid.layers.reduce_sum(node, dim=-1)
        return log_joint_

    def forward(self, x):
        nodes_q = self.variational({'x': x})
        z, logqz = nodes_q['z']
        nodes_p = self.generator({'x': x, 'z': z})
        logpxz = self.log_joint(nodes_p)
        elbo = fluid.layers.reduce_mean(logpxz - logqz)
        return -elbo

