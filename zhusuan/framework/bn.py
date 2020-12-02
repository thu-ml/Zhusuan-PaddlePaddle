""" Bayesian Network """

import numpy as np
import paddle

import paddle.fluid as fluid

from zhusuan.framework.stochastic_tensor import StochasticTensor
from zhusuan.distributions import *

class BayesianNet(paddle.nn.Layer):
    """
    A Bayeisian Network models a generative process for certain varaiables: p(x,z|y) or p(z|x,y) or p(x|z,y)
    """

    def __init__(self, observed=None):
        super(BayesianNet, self).__init__()
        self._nodes = {}
        self._cache = {}
        self._observed = observed if observed else {}

    @property
    def nodes(self):
        return self._nodes

    @property
    def cache(self):
        return self._cache

    @property
    def observed(self):
        return self._observed

    def observe(self, observed):
        self._observed = {}
        for k,v in observed.items():
            self._observed[k] = v


    def sn(self, *args, **kwargs):
        ## short cut for self stochastic_node
        return self.stochastic_node(*args, **kwargs)

    def stochastic_node(self,
                        distribution,
                        name,
                        **kwargs):
        _dist = globals()[distribution](**kwargs)
        self._nodes[name] = StochasticTensor(self, name, _dist, **kwargs)
        return self._nodes[name].tensor

    def _log_joint(self):
        _ret = 0
        for k,v in self._nodes.items():
            if isinstance(v, StochasticTensor):
                try:
                    _ret = _ret + v.log_prob()
                except:
                    _ret = v.log_prob()
        return _ret

    def log_joint(self, use_cache=False):
        """
        The default log joint probability of this :class:`BayesianNet`.
        It works by summing over all the conditional log probabilities of
        stochastic nodes evaluated at their current values (samples or
        observations).
        :return: A Tensor.
        """
        if use_cache:
            if not hasattr(self, "_log_joint_cache"):
                self._log_joint_cache = self._log_joint()
        else:
            self._log_joint_cache = self._log_joint()

        return self._log_joint_cache

    def __getitem__(self, name):
        name = self._check_name_exist(name)
        return self._nodes[name]

