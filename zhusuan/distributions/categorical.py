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
        self._n_categories = kwargs['n_categories']

    @property
    def probs(self):
        """The un-normalized probabilities."""
        return self._probs

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories


    def _get_batch_shape(self):
        """
        Private method for subclasses to rewrite the :meth:`get_batch_shape`
        method.
        """
        return self.probs.shape[:-1]

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
            # Output shape: [ -1, n_samples]
            return sample_flat_

        sample_shape_ = np.concatenate([[n_samples], self.batch_shape], axis=0).tolist()
        sample_ = paddle.reshape(sample_flat_, sample_shape_)
        self.sample_cache = sample_
        assert (sample_.shape[0] == n_samples)
        # Output shape: [ batch_shape..., n_samples]
        return sample_

    def _log_prob(self, sample=None):

        if sample is None:
            sample = self.sample_cache

        if len(sample.shape) > len(self._probs.shape):
            _probs = paddle.tile(self._probs, repeat_times=\
                        [sample.shape[0], *len(self._probs.shape)*[1]])
        else:
            _probs = self._probs

        # `labels` type to calculate `sparse_softmax_cross_entropy_with_logits` must be
        # int32 or int64
        if self.dtype == 'float32':
            sample = paddle.cast(sample, 'int32')
        elif self.dtype == 'float64':
            sample = paddle.cast(sample, 'int64')

        ## Log Prob
        _probs = paddle.nn.functional.softmax(_probs)
        log_prob = fluid.layers.reduce_sum(sample * paddle.log(_probs + 1e-8), dim=-1)
        # [Notification]: softmax_cross_entropy_with_logits equals to:
        #   - fluid.layers.reduce_sum(sample * paddle.log(logits), axis=1)
        return log_prob


Discrete = Categorical