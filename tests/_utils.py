import torch as T
import numpy as np
from tensor import Tensor


def check(a1: T.Tensor | np.typing.NDArray, a2: Tensor, rtol=1e-05, atol=1e-08):
    assert isinstance(a1, (T.Tensor, np.ndarray)), type(a1)
    assert isinstance(a2, Tensor), type(a2)

    # assert a1.stride() == a2.stride, (a1.stride(), a2.stride)

    a1_numpy = a1.numpy(force=True) if isinstance(a1, T.Tensor)else a1
    a2_numpy = a2.numpy()

    # if contiguous:
    #     a1_numpy = np.ascontiguousarray(a1_numpy)
    #     a2_numpy = np.ascontiguousarray(a2_numpy)

    assert a1_numpy.shape == a2_numpy.shape, (a1_numpy.shape, a2_numpy.shape)
    assert a1_numpy.dtype == a2_numpy.dtype, (a1_numpy.dtype, a2_numpy.dtype)
    assert a1_numpy.strides == a2_numpy.strides, (
        a1_numpy.strides, a2_numpy.strides)

    if not np.allclose(a1_numpy, a2_numpy, rtol=rtol, atol=atol):
        raise Exception(
            f"mean_diff: {np.abs(a1_numpy - a2_numpy).mean()}\n"
            f"expected : {a1_numpy.flatten()[:10]}\n"
            f"found    : {a2_numpy.flatten()[:10]}\n"
        )


def from_torch(a: T.Tensor): return Tensor.from_numpy(a.numpy())
def from_numpy(a: np.typing.NDArray): return Tensor.from_numpy(a)
