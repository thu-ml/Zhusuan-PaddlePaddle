#
import paddle
import paddle.fluid as fluid
from zhusuan import distributions

__all__ = [
    'StochasticTensor',
]


class StochasticTensor(object):
    """
    The :class:`StochasticTensor` class represents the stochastic nodes in a
    :class:`BayesianNet`.
    We can use any distribution available in :mod:`zhusuan.distributions` to
    construct a stochastic node in a :class:`BayesianNet`. For example::
        bn = zs.BayesianNet()
        x = bn.normal("x", 0., std=1.)
    will build a stochastic node in ``bn`` with the
    :class:`~zhusuan.distributions.univariate.Normal` distribution. The
    returned ``x`` will be a :class:`StochasticTensor`. The second line is
    equivalent to::
        dist = zs.distributions.Normal(0., std=1.)
        x = bn.stochastic("x", dist)
    :class:`StochasticTensor` instances are Tensor-like, which means that
    they can be passed into any Tensorflow operations. This makes it easy
    to build Bayesian networks by mixing stochastic nodes and Tensorflow
    primitives.
    .. seealso::
        For more information, please refer to :doc:`/tutorials/concepts`.
    :param bn: A :class:`BayesianNet`.
    :param name: A string. The name of the :class:`StochasticTensor`. Must be
        unique in a :class:`BayesianNet`.
    :param dist: A :class:`~zhusuan.distributions.base.Distribution`
        instance that determines the distribution used in this stochastic node.
    :param observation: A Tensor, which matches the shape of `dist`. If
        specified, then the :class:`StochasticTensor` is observed and
        the :attr:`tensor` property will return the `observation`. This
        argument will overwrite the observation provided in
        :meth:`zhusuan.framework.meta_bn.MetaBayesianNet.observe`.
    :param n_samples: A 0-D `int32` Tensor. Number of samples generated by
        this :class:`StochasticTensor`.
    """

    def __init__(self, bn, name, dist, observation=None, **kwargs):
        if bn is None:
            pass
        self._bn = bn

        self._name = name
        self._dist = dist
        self._dtype = dist.dtype
        #print(kwargs)
        self._n_samples = kwargs.get("n_samples", None)
        self._observation = observation
        super(StochasticTensor, self).__init__()

        ## when computing log_prob, which dims are averaged or summed
        self._reduce_mean_dims = kwargs.get("reduce_mean_dims", None)
        self._reduce_sum_dims = kwargs.get("reduce_sum_dims", None)
        self._multiplier = kwargs.get("multiplier", None)

    def _check_observation(self, observation):
        return observation

    @property
    def bn(self):
        """
        The :class:`BayesianNet` where the :class:`StochasticTensor` lives.
        :return: A :class:`BayesianNet` instance.
        """
        return self._bn

    @property
    def name(self):
        """
        The name of the :class:`StochasticTensor`.
        :return: A string.
        """
        return self._name

    @property
    def dtype(self):
        """
        The sample type of the :class:`StochasticTensor`.
        :return: A ``DType`` instance.
        """
        return self._dtype

    @property
    def dist(self):
        """
         The distribution followed by the :class:`StochasticTensor`.
        :return: A :class:`~zhusuan.distributions.base.Distribution` instance.
        """
        return self._dist

    def is_observed(self):
        """
        Whether the :class:`StochasticTensor` is observed or not.
        :return: A bool.
        """
        return self._observation is not None

    @property
    def tensor(self):
        """
        The value of this :class:`StochasticTensor`. If it is observed, then
        the observation is returned, otherwise samples are returned.
        :return: A Tensor.
        """
        if self._name in self._bn.observed.keys():
            self._dist.sample_cache = self._bn.observed[self._name]
            return self._bn.observed[self._name]
        else:
            _samples = self._dist.sample(n_samples=self._n_samples)
        return _samples

    @property
    def shape(self):
        """
        Return the static shape of this :class:`StochasticTensor`.
        :return: A ``TensorShape`` instance.
        """
        return self.tensor.shape

    def log_prob(self,sample=None, **kwargs):
        _log_probs = self._dist.log_prob(sample, **kwargs)

        if self._reduce_mean_dims:
            _log_probs = fluid.layers.reduce_mean(_log_probs, self._reduce_mean_dims, keep_dim=True)

        if self._reduce_sum_dims:
            _log_probs = fluid.layers.reduce_sum(_log_probs, self._reduce_sum_dims, keep_dim=True)

        if self._reduce_mean_dims or self._reduce_sum_dims:
            _m = self._reduce_mean_dims if self._reduce_mean_dims else []
            _s = self._reduce_sum_dims if self._reduce_sum_dims else []
            _log_probs = fluid.layers.squeeze(_log_probs, [*_m, *_s])

        if self._multiplier:
            _log_probs = _log_probs * self._multiplier

        return _log_probs

