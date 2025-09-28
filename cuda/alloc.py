

import ctypes
import numpy as np
from ctypes import c_int, c_void_p, POINTER, c_size_t, byref
from .utils import _define_func, assert_cuda_error


# for the compiling and running? runtimes to be the same
_cuda = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)


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
