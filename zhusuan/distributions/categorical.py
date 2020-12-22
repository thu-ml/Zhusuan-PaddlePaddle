import paddle
import paddle.fluid as fluid

import numpy as np

from .base import Distribution

__all__ = [
    'Categorical',
    'Discrete'
]


class Categorical(Distribution):
    def __init__(self,
                 dtype='int32',
                 param_dtype='float32',
                 is_continues=False,
                 is_reparameterized=True,
                 group_ndims=0,
                 **kwargs):
        super(Categorical, self).__init__(dtype,
                             param_dtype,
                             is_continues,
                             is_reparameterized,
                             group_ndims=group_ndims,
                             **kwargs)

        self._probs = kwargs['probs']
        self._n_categories = self._probs.shape[-1]

    @property
    def probs(self):
        """The un-normalized probabilities."""
        return self._probs

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the :attr:`batch_shape`
        property.
        """
        return self.probs.shape[:-1] #if len(self.probs.shape) > 1 else [1]

    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        return self.probs.shape[:-1] #if len(self.probs.shape) > 1 else [1]

    def _sample(self, n_samples=1, **kwargs):

        if len(self._probs.shape) == 2:
            probs_flat = self._probs
        else:
            probs_flat = paddle.reshape(self._probs, [-1, self._n_categories])

        cate_ = paddle.distribution.Categorical(probs_flat)
        sample_flat_ = paddle.cast(cate_.sample([n_samples]), self.dtype)
        # samples_flat =  paddle.cast( paddle.transpose(cat.sample([n_samples]),
        #                                               np.arange(len(probs_flat.shape)-1,-1,-1).tolist()), self.dtype)

        if len(self._probs.shape) == 2:
            self.sample_cache = sample_flat_
            # Output shape: [ -1, n_samples]
            return sample_flat_

        if self.batch_shape in [ [] ]:
            sample_shape_ = [n_samples]
        else:
            sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        sample_ = paddle.reshape(sample_flat_, sample_shape_)

        self.sample_cache = sample_
        assert (sample_.shape[0] == n_samples)
        # Output shape: [ batch_shape..., n_samples]
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        ## Log Prob
        # TODO: Paddle do not have sparse_softmax_cross_entropy_with_logits,
        #  should check if it equals to the equations below:

        normalized_logits = self.probs - paddle.logsumexp( self.probs, axis=-1, keepdim=True)
        one_hot_ = paddle.nn.functional.one_hot(paddle.to_tensor(sample), self.probs.shape[-1])
        log_prob = fluid.layers.reduce_sum(one_hot_ * normalized_logits, dim=-1)

        # # A different way to calculate log_prob:
        # log_prob = -fluid.layers.reduce_sum(paddle.nn.functional.softmax_with_cross_entropy(
        #     label=sample, logits=_probs, soft_label=True), dim=-1)

        return log_prob





Discrete = Categorical