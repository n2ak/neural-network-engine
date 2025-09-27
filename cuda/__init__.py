
import subprocess
import ctypes
import tempfile
import os
import numpy as np
from ctypes import c_int, c_void_p, POINTER, c_size_t, byref
from ._bin_ops import *
from ._unary_ops import *
from ._reduce_ops import *
from ._elemwsie_ops import *
from .utils import _define_func, assert_cuda_error

MAX_DIMS = 4
HEADER = f"""
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_DIMS {MAX_DIMS}
#define float32 float
#define float64 double
#define int32 int
#define int64 long long
#define sum(a,b) a+b

#define _add(a,b) a+b
#define _mul(a,b) a*b
#define _sub(a,b) a-b
#define _div(a,b) a/b

__device__ inline void unravel_index(int idx, const int* shape, int ndim, int* coords) {{
    for (int d = ndim - 1; d >= 0; --d) {{
        coords[d] = idx % shape[d];
        idx /= shape[d];
    }}
}}

inline int _size(const int *shape,int ndim){{
    int size = 1;
    for(int i = 0;i < ndim; i++){{
        size *= shape[i];
    }}
    return size;
}}

void shapeToDevice(const int *shape,int **d_shape,int ndim){{
    cudaMalloc(d_shape, ndim * sizeof(int));
    cudaMemcpy(*d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);
}}

__device__ inline int flattenIndex(int ndim,const int* coords,const int* stride){{
    int flat = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        flat += coords[d] * stride[d];
    }}
    return flat;
}}
"""


def compile_cuda(code: str, lib_name="libtemp.so"):
    tmpdir = tempfile.mkdtemp()
    cu_path = os.path.join(tmpdir, "kernel.cu")
    so_path = os.path.join(tmpdir, lib_name)
    with open(cu_path, "w") as f:
        f.write(code)

    cmd = [
        nvcc_path, "-shared", "-Xcompiler", "-fPIC", cu_path,
        "-o", so_path,
        "-lcudart",
        "-arch=sm_86",
    ]
    subprocess.check_call(cmd)
    lib = ctypes.CDLL(so_path, mode=RTLD_GLOBAL)
    return lib


# for the compiling and running? runtimes to be the same
RTLD_GLOBAL = ctypes.RTLD_GLOBAL

nvcc_path = "/usr/local/cuda-12.8/bin/nvcc"
_cuda = ctypes.CDLL("libcudart.so", mode=RTLD_GLOBAL)


def get_cuda_code():
    matmul = matmul_code()
    uops = "\n\n".join(map(lambda v: uops_code(*v), UOPS))
    bin_op = "\n\n".join(map(lambda v: bin_ops_code(*v), BIN_OPS))
    recudeops = "\n\n".join(map(lambda v: recude_code(*v), REDUCE_OPS))
    elemwise_ops = "\n\n".join(map(lambda v: elemwise_code(*v), ELEMWISE_OPS))

    return "\n".join([
        HEADER,
        matmul,
        recudeops,
        elemwise_ops,
        bin_op,
        uops,
    ])


class Buffer:
    def __init__(self, nbytes: int, allocated=False):
        ptr = c_void_p()
        err = CudaAllocator.alloc(byref(ptr), nbytes)
        assert_cuda_error(err)
        self.ptr = ptr
        self.nbytes = nbytes
        self.allocated = allocated
        self.refcount = 1

    def free(self):
        self.refcount -= 1
        if self.refcount <= 0:
            CudaAllocator._free(self.ptr)
            CudaAllocator.mem -= self.nbytes

    def ref(self):
        self.refcount += 1
        return self


class CudaAllocator:
    _cudaMemcpyHostToDevice = 1
    _cudaMemcpyDeviceToHost = 2
    alloc = _define_func(_cuda.cudaMalloc, [
                         POINTER(c_void_p), c_size_t], c_int)
    _free = _define_func(_cuda.cudaFree, [c_void_p], c_int)
    memcpy = _define_func(
        _cuda.cudaMemcpy, [c_void_p, c_void_p, c_size_t, c_int], c_int)
    get_last_error = _define_func(_cuda.cudaGetLastError, [], c_int)
    _sync = _define_func(_cuda.cudaDeviceSynchronize, [], c_int)

    mem = 0

    @classmethod
    def alloc_empty(cls, shape, dtype: np.typing.DTypeLike) -> Buffer:
        buffer = Buffer(np.dtype(dtype).itemsize *
                        np.prod(shape, dtype=int), allocated=True)
        # print(f"Allocated {buffer.nbytes} bytes")
        cls.mem += buffer.nbytes
        return buffer

    @classmethod
    def to_cuda(cls, host_array: np.typing.NDArray):
        buffer = cls.alloc_empty(host_array.shape, host_array.dtype)
        err = cls.memcpy(
            buffer.ptr,
            host_array.ctypes.data_as(c_void_p),
            host_array.nbytes,
            cls._cudaMemcpyHostToDevice
        )
        assert_cuda_error(err)
        return buffer

    @classmethod
    def from_cuda(cls, buffer: Buffer, shape, dtype, stride):
        assert buffer.refcount > 0
        assert isinstance(buffer, Buffer)
        assert buffer.allocated

        arr = np.empty(shape, dtype=dtype)

        nbytes = get_nbytes(shape, stride, dtype)
        # TODO: do this only if stride and shape dont correspond
        arr = np.lib.stride_tricks.as_strided(
            arr, shape=shape, strides=[s * arr.itemsize for s in stride]
        )

        cls.synchronize()
        err = cls.memcpy(
            arr.ctypes.data_as(c_void_p),
            buffer.ptr,
            nbytes,
            cls._cudaMemcpyDeviceToHost
        )
        assert_cuda_error(err)
        return arr

    @classmethod
    def synchronize(cls,):
        assert_cuda_error(cls.get_last_error())
        assert_cuda_error(cls._sync())

    @classmethod
    def free(cls, buffer: Buffer):
        buffer.free()


def get_nbytes(shape, stride, dtype):
    itemsize = np.dtype(dtype).itemsize
    size = 1
    for i in range(len(shape)):
        if stride[i] == 0:
            continue
        size *= shape[i]
    return size * itemsize


lib = compile_cuda(get_cuda_code(), "cuda_code")


def register_ops():
    ops = {}
    register_elemwise_ops(lib, ops)
    register_bin_ops(lib, ops)
    register_uops(lib, ops)
    register_reduce_ops(lib, ops)
    return ops


_cuda_ops = register_ops()
