from collections import namedtuple
import paddle

__all__ = [
    "SGLD",
]

class SGLD(SGMCMC):
    """
    Subclass of SGMCMC which implements Stochastic Gradient Langevin Dynamics
    (Welling & Teh, 2011) (SGLD) update. The updating equation implemented
    below follows Equation (3) in the paper.
    Attributes of returned `sgmcmc_info` in :meth:`SGMCMC.sample`:
    * **q** - The updated values of latent variables.
    :param learning_rate: A 0-D `float32` Tensor. It can be either a constant
        or a placeholder for decaying learning rate.
    """
    def __init__(self, learning_rate):
        self.lr = tf.convert_to_tensor(
            learning_rate, tf.float32, name="learning_rate")
        super(SGLD, self).__init__()

    def _define_variables(self, qs):
        pass

    def _update(self, qs, grad_func):
        return zip(*[self._update_single(q, grad)
                     for q, grad in zip(qs, grad_func(qs))])

    def _update_single(self, q, grad):
        new_q = q + 0.5 * self.lr * grad + tf.random_normal(
            tf.shape(q), stddev=tf.sqrt(self.lr))
        update_q = q.assign(new_q)
        info = {"q": new_q}
        return update_q, info

