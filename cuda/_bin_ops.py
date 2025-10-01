
from ctypes import c_int, c_void_p, POINTER
from .utils import _define_func, _int_1d_array


def define_matmul(lib):
    return _define_func(lib["matmul_batched"], [
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
        name = f"copy_out_indices_{dtype}"
        ops[name] = _define_func(lib[name], [
            # const T * d_A, const int * shape_A, const int * stride_A,
            c_void_p,  _int_1d_array(), _int_1d_array(),
            # T * d_C, const int ** indices, const int * shape_C,
            c_void_p, POINTER(c_void_p), _int_1d_array(),
            # int ndim_A
            c_int,
        ], None)
