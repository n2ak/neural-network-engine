
import numpy as np


def dataloader(*arrs: np.typing.NDArray, batch_size=64, shuffle=True):
    idx = 0
    dlen = arrs[0].shape[0]
    if shuffle:
        perm = np.random.permutation(dlen)
        arrs = tuple(arr[perm] for arr in arrs)

    while idx < dlen:
        yield tuple(arr[idx:idx+batch_size] for arr in arrs)
        idx += batch_size
