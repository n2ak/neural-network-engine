
import numpy as np
from typing import Optional
from ctypes import c_int, c_void_p, POINTER, c_size_t, byref

from . import Binary
from .utils import assert_cuda_error


# for the compiling and running? runtimes to be the same
_CUDA = Binary("libcudart.so")
MAX_DIMS = 4


class Buffer:
    @property
    def nbytes(self): return get_nbytes(self.shape, self.stride, self.dtype)

    def __init__(self, ptr: c_void_p, shape: tuple[int, ...], stride: tuple[int, ...], dtype: np.typing.DTypeLike, view=False, parent: Optional["Buffer"] = None, slice: Optional[tuple[slice, ...]] = None):
        assert isinstance(shape, (tuple, list)), (type(shape), shape)

        assert dtype is not None
        assert shape is not None
        assert stride is not None

        if slice is not None:
            assert view
        if view:
            assert parent is not None

        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.dtype = np.dtype(dtype)

        self.ptr = ptr
        self.refcount = 1
        self.is_view = view
        self.parent = parent
        self._slice = slice

    @classmethod
    def new(cls, shape: tuple[int, ...], stride: tuple[int, ...], dtype: np.typing.DTypeLike):
        ptr = c_void_p()
        assert_cuda_error(CudaAllocator.alloc.call(
            byref(ptr), get_nbytes(shape, stride, dtype))
        )
        buff = Buffer(
            ptr,
            shape=shape,
            stride=stride,
            dtype=dtype
        )
        return buff

    def free(self):
        if self.is_view:
            return
        self.refcount -= 1
        if self.refcount <= 0:
            CudaAllocator._free.call(self.ptr)
            CudaAllocator.mem -= self.nbytes

    def _as_view(self, shape, stride, offset, slice):  # offset in items
        self.refcount += 1
        ptr_offset = c_void_p(self.ptr.value + offset*self.dtype.itemsize)

        parent = self
        assert parent is not None

        buff = Buffer(
            ptr=ptr_offset,
            shape=shape,
            stride=stride,
            dtype=self.dtype,
            view=True,
            parent=parent,
            slice=slice
        )
        assert buff.nbytes <= buff.parent.nbytes  # type: ignore
        return buff

    def numpy(self):
        if self.is_view:
            assert self.parent is not None
            if self._slice is not None:
                return self.parent.numpy()[self._slice]
            parent = self.parent
            while parent.parent is not None:
                # get the orginal buffer who is not a view
                parent = parent.parent
            data = CudaAllocator.from_cuda(
                parent, new_shape=self.shape, new_stride=self.stride)
        else:
            data = CudaAllocator.from_cuda(self)
        assert isinstance(data, np.ndarray)
        return data


class CudaAllocator:
    _cudaMemcpyHostToDevice = 1
    _cudaMemcpyDeviceToHost = 2
    alloc = _CUDA.define_function(
        "cudaMalloc",
        [POINTER(c_void_p), c_size_t], c_int
    )
    _free = _CUDA.define_function("cudaFree", [c_void_p], c_int)
    memcpy = _CUDA.define_function(
        "cudaMemcpy",
        [c_void_p, c_void_p, c_size_t, c_int],
        c_int
    )
    get_last_error = _CUDA.define_function(
        "cudaGetLastError",
        [],
        c_int
    )
    sync = _CUDA.define_function(
        "cudaDeviceSynchronize",
        [],
        c_int
    )

    mem = 0

    @classmethod
    def alloc_empty(cls, shape, stride, dtype: np.typing.DTypeLike) -> Buffer:
        assert len(
            shape) < MAX_DIMS, "Only tensors of up to {MAX_DIMS} dimensions can be allocated!"
        buffer = Buffer.new(shape, stride, dtype)
        # print(f"Allocated {buffer.nbytes} bytes")
        cls.mem += buffer.nbytes
        return buffer

    @classmethod
    def to_cuda(cls, host_array: np.typing.NDArray):
        buffer = cls.alloc_empty(host_array.shape, [
                                 s//host_array.itemsize for s in host_array.strides], host_array.dtype)
        err = cls.memcpy.call(
            buffer.ptr,
            host_array.ctypes.data_as(c_void_p),
            host_array.nbytes,
            cls._cudaMemcpyHostToDevice
        )
        assert_cuda_error(err)
        return buffer

    @classmethod
    def from_cuda(cls, buffer: Buffer, new_shape=None, new_stride=None):
        # shape, dtype, stride = buffer.shape, buffer.dtype, buffer.stride
        assert buffer.refcount > 0
        assert isinstance(buffer, Buffer)
        assert not buffer.is_view

        arr = np.empty(buffer.shape, dtype=buffer.dtype)

        # TODO: do this only if not contiguous
        cls.synchronize()
        assert_cuda_error(cls.memcpy.call(
            arr.ctypes.data_as(c_void_p),
            buffer.ptr,
            buffer.nbytes,
            cls._cudaMemcpyDeviceToHost
        ))
        if new_shape is None and new_stride is None:
            new_shape = buffer.shape
            new_stride = buffer.stride
        arr = np.lib.stride_tricks.as_strided(
            arr, shape=new_shape, strides=[
                s * arr.itemsize for s in new_stride]  # type: ignore
        )
        return arr

    @classmethod
    def synchronize(cls,):
        assert_cuda_error(cls.get_last_error.call())
        assert_cuda_error(cls.sync.call())

    @classmethod
    def free(cls, buffer: Buffer):
        buffer.free()


def get_nbytes(shape: tuple[int, ...], stride: tuple[int, ...], dtype: np.typing.DTypeLike):
    itemsize = np.dtype(dtype).itemsize
    size = 1
    for i in range(len(shape)):
        if stride[i] == 0:
            continue
        size *= shape[i]
    return size * itemsize
