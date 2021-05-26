import paddle
import numpy as np

def get_coupling_mask(n_dim, n_channel, n_mask, split_type='OddEven'):
    '''

    Args:
        n_dim:
        n_channel:
        n_mask:
        split_type:

    Returns: List of generated mask

    '''
    masks = []
    if split_type == 'OddEven':
        if n_channel == 1:
            mask = paddle.arange(n_dim, dtype='float32') % 2
            for i in range(n_mask):
                masks.append(mask)
                mask = 1. - mask
    elif split_type == 'Half':
        pass
    elif split_type == 'RandomHalf':
        pass
    else:
        raise NotImplementedError()
    return masks

k = get_coupling_mask(10, 1, 10)
for i in k:
    print(i.numpy())