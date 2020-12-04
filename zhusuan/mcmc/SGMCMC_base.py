from collections import namedtuple
import paddle

__all__ = [
    "SGMCMC",
]


class SGMCMC(object):
    """
    Base class for stochastic gradient MCMC (SGMCMC) algorithms.
    SGMCMC is a class of MCMC algorithms which utilize stochastic gradients
    instead of the true gradients. To deal with the problems brought by
    stochasticity in gradients, more sophisticated updating scheme, such as
    SGHMC and SGNHT, were proposed. We provided four SGMCMC algorithms here:
    SGLD, PSGLD, SGHMC and SGNHT. For SGHMC and SGNHT, we support 2nd-order
    integrators introduced in (Chen et al., 2015).
    The implementation framework is similar to that of
    :class:`~zhusuan.hmc.HMC` class. However, SGMCMC algorithms do not include
    Metropolis update, and typically do not include hyperparameter adaptation.
    The usage is the same as that of :class:`~zhusuan.hmc.HMC` class.
    Running multiple SGMCMC chains in parallel is supported.
    To use the sampler, the user first defines the sampling method and
    corresponding hyperparameters by calling the subclass :class:`SGLD`,
    :class:`PSGLD`, :class:`SGHMC` or :class:`SGNHT`. Then the user creates a
    (list of) tensorflow `Variable` storing the initial sample, whose shape is
    ``chain axes + data axes``. There can be arbitrary number of chain axes
    followed by arbitrary number of data axes. Then the user provides a
    `log_joint` function which returns a tensor of shape ``chain axes``, which
    is the log joint density for each chain. Alternatively, the user can also
    provide a `meta_bn` instance as a description of `log_joint`. Then the user
    runs the operation returned by :meth:`sample`, which updates the sample
    stored in the `Variable`.
    The typical code for SGMCMC inference is like::
        sgmcmc = zs.SGHMC(learning_rate=2e-6, friction=0.2,
                          n_iter_resample_v=1000, second_order=True)
        sample_op, sgmcmc_info = sgmcmc.make_grad_func(meta_bn,
            observed={'x': x, 'y': y}, latent={'w1': w1, 'w2': w2})
        with tf.Session() as sess:
            for _ in range(n_iters):
                _, info = sess.run([sample_op, sgmcmc_info],
                                      feed_dict=...)
                print("mean_k", info["mean_k"])   # For SGHMC and SGNHT,
                                                  # optional
    After getting the sample_op, the user can feed mini-batches to a data
    placeholder `observed` so that the gradient is a stochastic gradient. Then
    the user runs the sample_op like using HMC.
    """
    def __init__(self):
        self.t = tf.Variable(0, name="t", trainable=False, dtype=tf.int32)

    def _make_grad_func(self, meta_bn, observed, latent):
        if callable(meta_bn):
            self._log_joint = meta_bn
        else:
            self._log_joint = lambda obs: meta_bn.observe(**obs).log_joint()

        self._observed = observed
        self._latent = latent

        latent_k, latent_v = [list(i) for i in zip(*six.iteritems(latent))]
        for i, v in enumerate(latent_v):
            if not isinstance(v, tf.Variable):
                raise TypeError("latent['{}'] is not a tensorflow Variable."
                                .format(latent_k[i]))
        self._latent_k = latent_k
        self._var_list = latent_v

        def _get_log_posterior(var_list, observed):
            joint_obs = merge_dicts(dict(zip(latent_k, var_list)), observed)
            return self._log_joint(joint_obs)

        def _get_gradient(var_list, observed):
            return tf.gradients(
                _get_log_posterior(var_list, observed), var_list)

        return lambda var_list: _get_gradient(var_list, observed)

    def _apply_updates(self, grad_func):
        qs = self._var_list
        self._define_variables(qs)
        update_ops, infos = self._update(qs, grad_func)

        with tf.control_dependencies([self.t.assign_add(1)]):
            sample_op = tf.group(*update_ops)
        list_attrib = zip(*map(lambda d: six.itervalues(d), infos))
        list_attrib_with_k = map(lambda l: dict(zip(self._latent_k, l)),
                                 list_attrib)
        attrib_names = list(six.iterkeys(infos[0]))
        dict_info = dict(zip(attrib_names, list_attrib_with_k))
        SGMCMCInfo = namedtuple("SGMCMCInfo", attrib_names)
        sgmcmc_info = SGMCMCInfo(**dict_info)

        return sample_op, sgmcmc_info

    def sample(self, meta_bn, observed, latent):
        """
        Return the sampling `Operation` that runs a SGMCMC iteration and the
        statistics collected during it, given the log joint function (or a
        :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance), observed
        values and latent variables.
        :param meta_bn: A function or a
            :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance. If it
            is a function, it accepts a dictionary argument of ``(string,
            Tensor)`` pairs, which are mappings from all `StochasticTensor`
            names in the model to their observed values. The function should
            return a Tensor, representing the log joint likelihood of the
            model. More conveniently, the user can also provide a
            :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance
            instead of directly providing a log_joint function. Then a
            log_joint function will be created so that `log_joint(obs) =
            meta_bn.observe(**obs).log_joint()`.
        :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping
            from names of observed `StochasticTensor` s to their values.
        :param latent: A dictionary of ``(string, Variable)`` pairs.
            Mapping from names of latent `StochasticTensor` s to corresponding
            tensorflow `Variables` for storing their initial values and
            samples.
        :return: A Tensorflow `Operation` that runs a SGMCMC iteration, called
            `sample_op`.
        :return: A namedtuple that records some useful values, called
            `sgmcmc_info`. Suppose the list of keys of `latent` dictionary is
            ``['w1', 'w2']``. Then the typical structure of `sgmcmc_info` is
            ``SGMCMCInfo(attr1={'w1': some value, 'w2': some value},
            attr2={'w1': some value, 'w2': some value}, ...)``. Hence,
            ``sgmcmc_info.attr1`` is a dictionary containing the quantity
            `attr1` corresponding to each latent variable in the `latent`
            dictionary.
            `sgmcmc_info` returned by any SGMCMC algorithm has an attribute
            `q`, representing the updated values of latent variables. To check
            out other attributes, see the documentation for the specific
            subclass below.
        """
        grad_func = self._make_grad_func(meta_bn, observed, latent)
        return self._apply_updates(grad_func)

    def _update(self, qs, grad_func):
        return NotImplementedError()

    def _define_variables(self, qs):
        return NotImplementedError()
