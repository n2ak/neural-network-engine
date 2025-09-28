
from ctypes import c_int, c_void_p
from .utils import _define_func, _int_1d_array


def define_matmul(lib):
    return _define_func(lib["matmul_3D_2d"], [
        c_void_p,  # A
        c_void_p,  # B
        c_void_p,  # C
        _int_1d_array(),  # stride_A
        _int_1d_array(),  # stride_B
        _int_1d_array(),  # stride_C
        c_int, c_int, c_int, c_int,  # int BATCH, int M, int K, int N,
    ], None)


def define_copyout(lib, ops):
    for dtype in ["float32", "float64", "int32", "int64"]:
        name = f"copy_out_{dtype}"
        ops[name] = _define_func(lib[name], [
            c_void_p,  _int_1d_array(), _int_1d_array(),
            c_void_p,
            c_int,
        ], None)
