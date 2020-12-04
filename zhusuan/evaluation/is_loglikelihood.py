from zhusuan.variational import ImportanceWeightedObjective

all__ = [
    'is_loglikelihood',
]


def is_loglikelihood(bn, proposal, observed, axis=None):
    """
    Marginal log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.
    :param bn: A :class:`~zhusuan.framework.bn.BayesianNet`
        instance or a log joint probability function.
        For the latter, it must accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        node names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param proposal: A :class:`~zhusuan.framework.bn.BayesianNet` instance
        that defines the proposal distributions of latent nodes.
        `proposal` and `latent` are mutually exclusive.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed stochastic nodes to their values.
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in the objective. If ``None``, no dimension is
        reduced.
    :return: A Tensor. The estimated log likelihood of observed data.
    """

    _model = ImportanceWeightedObjective(bn, proposal, axis=axis)
    _ret = _model.forward(observed)

    return _ret
    

	
