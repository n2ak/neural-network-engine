from ctypes import c_int, c_void_p
from .utils import _int_1d_array


def define_matmul():
    from . import CUDA_KERNELS

    CUDA_KERNELS.define_function(
        "matmul_batched",
        [
            c_void_p,  # A
            c_void_p,  # B
            c_void_p,  # C
            _int_1d_array(),  # stride_A
            _int_1d_array(),  # stride_B
            _int_1d_array(),  # stride_C
            c_int,
            c_int,
            c_int,
            c_int,  # int BATCH, int M, int K, int N,
        ],
    )
