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
                    log_joint_ += nodes[n_name].log_prob()
                except:
                    log_joint_ = nodes[n_name].log_prob()
        else:
            log_joint_ = 0.
            for n_name in nodes.keys():
                prob_= fluid.layers.reduce_mean(nodes[n_name].log_prob(), dim=0)
                log_joint_ += fluid.layers.reduce_sum(prob_)

        return log_joint_

    def forward(self, observed, reduce_mean=True):
        nodes_q = self.variational(observed).nodes

        _v_inputs = {k:v.tensor for k,v in nodes_q.items()}
        _observed = {**_v_inputs, **observed}

        nodes_p = self.generator(_observed).nodes

        #print('observed.keys:', observed.keys())
        #print('_observed.keys:', _observed.keys())
        #print('nodes_p.keys: ', nodes_p.keys())
        #print('nodes_q.keys: ', nodes_q.keys())

        logpxz = self.log_joint(nodes_p, reduce_mean=reduce_mean)
        logqz = self.log_joint(nodes_q, reduce_mean=reduce_mean)
        elbo = fluid.layers.reduce_mean(logpxz - logqz)

        return -elbo

