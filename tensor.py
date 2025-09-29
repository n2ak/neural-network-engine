import os
import numpy as np
from typing import Self
from cuda import _cuda_ops
from cuda.alloc import CudaAllocator, Buffer
from cuda.utils import promote_dtype, promote_uop_dtype
from cuda.op_names import elemwise_op_name, uop_name, reduceop_name, reduction_op_name


def get_numpy_stride(arr: np.typing.NDArray):
    itemsize = arr.itemsize
    return tuple(s // itemsize for s in arr.strides)


def stride_from_shape(shape: tuple[int, ...] | list[int]):
    stride = []
    acc = 1
    for s in reversed(shape):
        stride.insert(0, acc)
        acc *= s
    return tuple(stride)


def as_tuple(input):
    return input if isinstance(input, (tuple, list)) else (input,)


class Tensor:

    def __init__(self, data: Buffer) -> None:
        self.data = data

    @property
    def shape(self): return self.data.shape
    @property
    def stride(self): return self.data.stride
    @property
    def dtype(self): return self.data.dtype

    @property
    def is_contiguous(self):
        stride = tuple(self.stride)
        expected_stride = stride_from_shape(self.shape)
        return expected_stride == stride

    @staticmethod
    def from_numpy(data: np.typing.NDArray):
        assert isinstance(data, np.ndarray), type(data)

        return Tensor(
            CudaAllocator.to_cuda(data),
        )

    @staticmethod
    def randn(*shape: int, dtype=np.float32):
        return Tensor.from_numpy(np.random.randn(*shape).astype(dtype))

    # def cpu(self) -> "Tensor":
    #     if self.is_cpu:
    #         return self
    #     if self.data.is_view:  # type: ignore
    #         assert self.data.parent is not None  # type: ignore
    #         dst = Tensor.empty(
    #             self.shape,
    #             dtype=self.dtype,
    #         )
    #         CUDA_OPS.copy_to(self.data, dst)  # type: ignore
    #         return dst.cpu()
    #     data = CudaAllocator.from_cuda(
    #         self.data, self.shape, self.dtype, stride=self.stride)  # type: ignore
    #     return Tensor(
    #         data,
    #     )

    @classmethod
    def empty(cls, shape, dtype: np.typing.DTypeLike = np.float32, ):
        data = CudaAllocator.alloc_empty(
            shape=shape, stride=stride_from_shape(shape), dtype=dtype)
        stride = stride_from_shape(shape)
        return Tensor(
            data,
        )

    @classmethod
    def zeros(cls, shape, dtype=np.float32):
        return Tensor.from_numpy(np.zeros(shape=shape, dtype=dtype))

    def astype(self, dtype) -> "Tensor":
        if dtype == self.dtype:
            return self
        # TODO: do it in gpu
        return Tensor.from_numpy(self.numpy().astype(dtype))

    def numpy(self) -> np.typing.NDArray:
        data = self.data.numpy()
        assert self.shape == data.shape, (self.shape, data.shape)
        return data

    def __del__(self):
        CudaAllocator.free(self.data)

    def __add__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("add", self, other)

    def __mul__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("mul", self, other)

    def __sub__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("sub", self, other)

    def __truediv__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("div", self, other, floating_op=True)

    def exp(self):
        return CUDA_OPS.uop("exp", self)

    def log(self):
        return CUDA_OPS.uop("log", self)

    def log2(self):
        return CUDA_OPS.uop("log2", self)

    def max(self, axis: int | tuple[int, ...] = (), keepdim=False):
        return CUDA_OPS.reduce_op("max", self,  axis=axis, keepdim=keepdim)

    def sum(self, axis: int | tuple[int, ...] = (), keepdim=False):
        out_dtype = self.dtype
        if self.dtype == np.int32:
            out_dtype = np.dtype(np.int64)
        return CUDA_OPS.reduce_op("sum", self, axis=axis, keepdim=keepdim, out_dtype=out_dtype)

    def mean(self, axis: int | tuple[int, ...] = (), keepdim=False):
        d = np.prod([self.shape[a] for a in self._correct_axis(axis)]).item()
        return self.sum(axis=axis, keepdim=keepdim) / d

    def _correct_axis(self, axis: int | tuple[int, ...]):
        axis = as_tuple(axis)
        if axis == ():
            axis = tuple(range(self.ndim))
        return axis

    @property
    def ndim(self): return len(self.shape)
    @property
    def size(self): return np.prod(self.shape, dtype=np.int32).item()

    def transpose(self, dim1: int, dim2: int):
        dims = list(range(self.ndim))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(
            *dims
        )

    def permute(self, *dims: int):
        new_stride = [self.stride[d] for d in dims]
        new_shape = [self.shape[d] for d in dims]
        return self._as_view(new_shape, new_stride)

    def contiguous(self):
        if self.is_contiguous:
            return self
        new_stride = stride_from_shape(self.shape)
        # TODO: do it in gpu
        return Tensor.from_numpy(np.ascontiguousarray(self.numpy()))

    def _broadcastable(self, other: Self):
        for d1, d2 in zip(self.shape[::-1], other.shape[::-1]):
            if d1 == 1 or d2 == 1:
                continue
            if d1 != d2:
                return False
        return True

    def try_broadcast(self, other: Self):
        if self.shape == other.shape:
            return self, other

        assert self._broadcastable(other)
        if self.ndim < other.ndim:
            pass
        a_shape = self.shape[::-1]
        b_shape = other.shape[::-1]

        expected_shape = [1] * max(self.ndim, other.ndim)

        for i in range(len(expected_shape)):
            d1 = a_shape[i] if i < self.ndim else -1
            d2 = b_shape[i] if i < other.ndim else -1
            expected_shape[i] = max(d1, d2)
        expected_shape = expected_shape[::-1]
        return self.expand(*expected_shape), other.expand(*expected_shape)

    def expand(self, *dims: int):
        if self.shape == dims:
            return self
        new_shape = dims
        expected_stride = [0] * len(dims)
        for i, d, sh, s in zip(range(self.ndim), dims[::-1], self.shape[::-1], self.stride[::-1]):
            if sh == d:
                expected_stride[i] = s
            else:
                if sh != 1:
                    raise Exception(f"dimension at {i} should be 1")
                expected_stride[i] = 0
        expected_stride = expected_stride[::-1]
        return self._as_view(new_shape, expected_stride)

    def _as_view(self, shape, stride, ptr_offset=0, slice=None):
        return Tensor(
            data=self.data._as_view(
                shape, stride, offset=ptr_offset, slice=slice),  # type: ignore
        )

    def __matmul__(self, other: Self):
        return CUDA_OPS.matmul(self, other)

    def __getitem__(self, keys):
        shape = self.shape
        stride = self.stride

        assert isinstance(keys, tuple), type(keys)
        assert all(map(lambda s: isinstance(s, slice), keys))
        if len(keys) < self.ndim:
            left = self.ndim - len(keys)
            keys = tuple(list(keys) + [slice(None) for _ in range(left)])
        assert len(keys) == self.ndim

        slices: np.ndarray = np.array([
            (s.start or 0, s.stop or d, s.step or 1)
            for d, s in zip(shape, keys)])
        # [30:...] and the dim is lower than 30 => we clip
        slices[:, 1] = np.clip(slices[:, 1], 0, shape)

        new_shape = np.ceil((slices[:, 1] - slices[:, 0]) / slices[:, 2])
        # clip negative dimensions
        new_shape = np.clip(new_shape, 0, shape)

        slices[:, 0] = np.clip(slices[:, 0], 0, np.array(shape))
        # slices[:, 0] = np.where(slices[:, 0] < shape, slices[:, 0], 0)

        offsets = (slices[:, 0] * stride)
        # strides at 0 dimensions shouldnt contribute to add offset
        # it doesn't matter because the size is 0, just to match numpy's results
        offsets = np.where(new_shape == 0, 0, offsets)
        offset = offsets.sum().item()

        new_stride = (slices[:, 2] * stride)
        # strides at 0 dimensions shouldnt add contribute to stride
        # it doesn't matter because the size is 0, just to match numpy's results
        new_stride = np.where(new_shape == 0, stride, new_stride)

        new_shape = tuple(new_shape.astype(int).tolist())
        new_stride = tuple(new_stride.astype(int).tolist())

        # print(shape, "shape->", new_shape)
        # print(stride, "stride->", new_stride)

        return self._as_view(
            shape=new_shape,
            stride=new_stride,
            ptr_offset=offset,
            slice=keys
        )


class CUDA_OPS:
    _kernels = _cuda_ops

    @classmethod
    def elem_op(cls, op_name, a: Tensor, b: Tensor | int | float, floating_op=False):
        assert isinstance(a, Tensor)
        if isinstance(b, (int, float)):
            b = Tensor.from_numpy(np.array(b, dtype=a.dtype))
            b = b.expand(*a.shape)

        out_dtype = np.dtype(promote_dtype(a.dtype, b.dtype, floating_op))
        kernel = cls._kernels[elemwise_op_name(
            op_name, a.dtype, b.dtype, out_dtype)]

        a, b = a.try_broadcast(b)
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)

        c = Tensor.empty(a.shape, dtype=out_dtype)

        a_stride = np.array(a.stride, dtype=np.int32)
        b_stride = np.array(b.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)

        kernel(
            a.data.ptr, a_stride,  # type: ignore
            b.data.ptr, b_stride,  # type: ignore
            c.data.ptr, c_stride,  # type: ignore
            shape,
            ndim
        )
        return c

    @classmethod
    def uop(cls, op_name, a: Tensor, floating_op=True):
        out_dtype = promote_uop_dtype(a.dtype, floating_op)
        c = Tensor.empty(a.shape, dtype=out_dtype)
        kernel = cls._kernels[uop_name(op_name, a.dtype, out_dtype)]
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)

        a_stride = np.array(a.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)
        kernel(
            a.data.ptr, a_stride,  # type: ignore
            c.data.ptr, c_stride,  # type: ignore
            shape,
            ndim
        )
        return c

    @classmethod
    def reduce_op(cls, op_name, a: Tensor, axis: int | tuple[int, ...], keepdim: bool, out_dtype=None):
        if out_dtype is None:
            out_dtype = a.dtype

        if axis == () and (os.getenv("USE_REDUCTION", "1") != "0"):
            # this is order of 1000s faster
            kernel = cls._kernels[reduction_op_name(
                op_name, str(a.dtype), str(out_dtype))]
            out = Tensor.empty((), dtype=out_dtype)
            kernel(
                a.data.ptr,  # type: ignore
                out.data.ptr,  # type: ignore
                a.size
            )
            return out
        axis = a._correct_axis(axis)

        def get_shape(shape: list[int]):
            if axis == ():
                return ()
            i = 0
            for a in axis:
                if keepdim:
                    shape[a] = 1
                else:
                    shape.pop(a-i)
                    i += 1
            return shape

        kernel = cls._kernels[reduceop_name(
            op_name, str(a.dtype), str(out_dtype))]
        c = Tensor.empty(get_shape(list(a.shape)), dtype=out_dtype)

        a_shape = np.array(a.shape, dtype=np.int32)
        c_shape = np.array(c.shape, dtype=np.int32)
        a_stride = np.array(a.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)

        kernel(
            a.data.ptr, a_stride, a_shape,  # type: ignore
            c.data.ptr, c_stride, c_shape,  # type: ignore
            np.array(axis, dtype=np.int32),
            a.ndim,
            c.ndim,
            len(axis),
            keepdim,
        )
        return c

    @classmethod
    def matmul(cls, a: Tensor, b: Tensor):
        assert a.ndim in [2, 3]
        assert b.ndim == 2
        assert a.shape[-1] == b.shape[0]
        assert a.dtype == b.dtype == np.float32

        K, N = b.shape
        if a.ndim == 3:
            BATCH, M, K = a.shape
            out = Tensor.empty(
                (BATCH, M, N),
                dtype=np.float32
            )
            a_stride = a.stride
            out_stride = out.stride
        else:
            BATCH = 1
            M, K = a.shape
            out = Tensor.empty(
                (M, N),
                dtype=np.float32
            )
            a_stride = [0] + list(a.stride)
            out_stride = [0] + list(out.stride)

        kernel = cls._kernels["matmul_3D_2d"]
        kernel(
            a.data.ptr,  # type: ignore
            b.data.ptr,  # type: ignore
            out.data.ptr,  # type: ignore
            np.array(a_stride, dtype=np.int32),
            np.array(b.stride, dtype=np.int32),
            np.array(out_stride, dtype=np.int32),
            BATCH, M, K, N,
        )
        return out

    @classmethod
    def copy_out(cls, src: Tensor, dst: Tensor):
        kernel_name = f"copy_out_{dst.dtype}"
        kernel = cls._kernels[kernel_name]

        src_shape = np.array(src.shape, dtype=np.int32)
        dst_shape = np.array(dst.shape, dtype=np.int32)
        src_stride = np.array(src.stride, dtype=np.int32)
        dst_stride = np.array(dst.stride, dtype=np.int32)

        kernel(
            src.data.ptr, src_shape, src_stride,  # type: ignore
            dst.data.ptr, dst_shape, dst_stride,  # type: ignore
            src.ndim,
            dst.ndim,
        )
        return dst

    @classmethod
    def copy_to(cls, data: Buffer, dst: Tensor):
        kernel_name = f"copy_out_{dst.dtype}"
        kernel = cls._kernels[kernel_name]

        src_shape = np.array(data.shape, dtype=np.int32)
        src_stride = np.array(data.stride, dtype=np.int32)

        # dst_shape = np.array(dst.shape, dtype=np.int32)
        # dst_stride = np.array(dst.stride, dtype=np.int32)

        kernel(
            data.ptr, src_shape, src_stride,  # type: ignore
            dst.data.ptr,   # type: ignore
            dst.ndim,
        )
        return dst
