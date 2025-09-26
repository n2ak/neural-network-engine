import torch as T
import numpy as np
from tensor import Tensor


def check(a1: T.Tensor, a2: Tensor):
    assert isinstance(a1, T.Tensor), type(a1)
    assert isinstance(a2, Tensor), type(a2)
    assert a1.stride() == a2.stride

    a1_numpy = a1.numpy()
    a2_numpy = a2.numpy()

    assert a1_numpy.shape == a2_numpy.shape
    assert a1_numpy.dtype == a2_numpy.dtype
    assert a1_numpy.strides == a2_numpy.strides
    if not np.allclose(a1_numpy, a2_numpy):
        raise Exception(
            f"mean_diff: {np.abs(a1_numpy - a2_numpy).mean()}\n"
            f"expected : {a1_numpy.flatten()[:10]}\n"
            f"found    : {a2_numpy.flatten()[:10]}\n"
        )


def from_torch(a: T.Tensor): return Tensor.from_numpy(a.numpy()).cuda()
