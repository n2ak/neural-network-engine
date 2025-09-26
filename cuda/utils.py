
import numpy as np


def _int_1d_array():
    return np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")


def _define_func(func, types, ret=None):
    func.argtypes = types
    func.restype = ret
    return func


def assert_cuda_error(err):
    assert err == 0, err


def promote_dtype(dtype1, dtype2, floating_op: bool) -> str:
    dt1 = np.dtype(dtype1)
    dt2 = np.dtype(dtype2)
    if np.issubdtype(dt1, np.floating) or np.issubdtype(dt2, np.floating):
        if dt1 == np.float64 or dt2 == np.float64:
            return "float64"
        return "float32"
    if floating_op:
        return "float32"
    if dt1 == np.int64 or dt2 == np.int64:
        return "int64"
    return dtype1 if dt1.itemsize >= dt2.itemsize else dtype2
