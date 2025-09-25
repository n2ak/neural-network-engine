from typing import Self
import numpy as np
from cuda import CudaAllocator, _cuda_ops, bin_op_name, elemwise_op_name, uop_name, Buffer


def get_numpy_stride(arr: np.typing.NDArray):
    itemsize = arr.itemsize
    return tuple(s // itemsize for s in arr.strides)


def stride_from_shape(shape: tuple[int] | list[int]):
    stride = []
    acc = 1
    for s in reversed(shape):
        stride.insert(0, acc)
        acc *= s
    return tuple(stride)


class Tensor:

    def __init__(self, data: Buffer | np.typing.NDArray, device, shape=None, dtype=None, stride=None) -> None:
        assert device in ["cuda", "cpu"]

        self.device = device
        if isinstance(data, np.ndarray):
            assert shape is None
            assert shape is dtype
            assert stride is dtype

            shape = data.shape
            dtype = data.dtype
            stride = get_numpy_stride(data)

        assert stride is not None
        assert shape is not None
        assert dtype is not None

        self.data = data
        self.shape: tuple[int] = tuple(shape)  # type: ignore
        self.stride: tuple[int] = tuple(stride)  # type: ignore
        self.dtype: np.typing.DTypeLike = dtype

    @property
    def is_contiguous(self):
        stride = tuple(self.stride)
        expected_stride = stride_from_shape(self.shape)
        return expected_stride == stride

    @staticmethod
    def from_numpy(data):
        assert isinstance(data, np.ndarray), type(data)
        return Tensor(data, device="cpu")

    def cuda(self):
        if self.is_cuda:
            return self
        data = CudaAllocator.to_cuda(self.data)  # type: ignore
        return Tensor(
            data,
            device="cuda",
            shape=self.shape,
            dtype=self.dtype,
            stride=self.stride,
        )

    def cpu(self):
        if self.is_cpu:
            return self
        data = CudaAllocator.from_cuda(
            self.data, self.shape, self.dtype, stride=self.stride)  # type: ignore
        return Tensor(
            data,
            device="cpu"
        )

    def to(self, device):
        if device == self.device:
            return self
        if device == "cpu":
            return self.cpu()
        if device == "cuda":
            return self.cuda()
        raise Exception(f"Invalid device {device}")

    @classmethod
    def empty(cls, shape, device="cpu", dtype: np.typing.DTypeLike = np.float32, ):
        stride = None
        if device == "cuda":
            data = CudaAllocator.alloc_empty(shape=shape, dtype=dtype)
            stride = stride_from_shape(shape)
        else:
            data = np.empty(shape, dtype=dtype)
        return Tensor(
            data,
            shape=shape,
            device=device,
            dtype=dtype,
            stride=stride,
        )

    @classmethod
    def zeros(cls, shape, device="cpu", dtype=np.float32):
        return Tensor.from_numpy(np.zeros(shape=shape, dtype=dtype)).to(device=device)

    def astype(self, dtype) -> "Tensor":
        if dtype == self.dtype:
            return self
        if self.is_cpu:
            data = self.data.astype(dtype)  # type: ignore
            return Tensor(
                data,
                device="cpu",
            )
        else:
            # TODO
            return self.cpu().astype(dtype).cuda()

    def numpy(self) -> np.typing.NDArray:
        data = self.cpu().data
        assert isinstance(data, np.ndarray)
        return data

    def __del__(self):
        if self.is_cuda:
            assert isinstance(self.data, Buffer), type(self.data)
            CudaAllocator.free(self.data)
        else:
            del self.data

    def __add__(self, other: Self | int | float):
        return CUDA_OPS.bin_op("add", self, other)

    def __mul__(self, other: Self | int | float):
        return CUDA_OPS.bin_op("mul", self, other)

    def __sub__(self, other: Self | int | float):
        return CUDA_OPS.bin_op("sub", self, other)

    def __truediv__(self, other: Self | int | float):
        return CUDA_OPS.bin_op("div", self, other)

    def exp(self):
        return CUDA_OPS.uop("exp", self)

    def log(self):
        return CUDA_OPS.uop("log", self)

    def log2(self):
        return CUDA_OPS.uop("log2", self)

    @property
    def ndim(self): return len(self.shape)
    @property
    def is_cuda(self): return self.device == "cuda"
    @property
    def is_cpu(self): return self.device == "cpu"

    def transpose(self, dim1: int, dim2: int):
        dims = list(range(self.ndim))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(
            *dims
        )

    def permute(self, *dims: int):
        assert self.is_cuda
        new_stride = [self.stride[d] for d in dims]
        new_shape = [self.shape[d] for d in dims]
        return Tensor(
            data=self.data.ref(),  # type: ignore
            dtype=self.dtype,
            shape=new_shape,
            stride=new_stride,
            device=self.device
        )

    def contiguous(self):
        assert self.is_cuda
        if self.is_contiguous:
            return self
        new_stride = stride_from_shape(self.shape)
        # TODO: do it in device
        return Tensor(
            data=np.ascontiguousarray(self.numpy()),
            device="cpu",
        ).to(self.device)

    def _broadcastable(self, other: Self):
        for d1, d2 in zip(self.shape[::-1], other.shape[::-1]):
            if d1 == 1 or d2 == 1:
                continue
            if d1 != d2:
                return False
        return True

    def try_broadcast(a, b: Self):
        assert a._broadcastable(b)
        if a.ndim < b.ndim:
            pass
        a_shape = a.shape[::-1]
        b_shape = b.shape[::-1]

        expected_shape = [1] * max(a.ndim, b.ndim)

        for i in range(len(expected_shape)):
            d1 = a_shape[i] if i < a.ndim else -1
            d2 = b_shape[i] if i < b.ndim else -1
            expected_shape[i] = max(d1, d2)
        expected_shape = expected_shape[::-1]
        return a.expand(*expected_shape), b.expand(*expected_shape)

    def expand(self, *dims: int):
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

        return Tensor(
            data=self.data.ref(),  # type: ignore
            dtype=self.dtype,
            shape=new_shape,
            stride=expected_stride,
            device=self.device
        )


class CUDA_OPS:
    _ops = _cuda_ops

    @classmethod
    def elem_op(cls, op_name, a: Tensor, b: Tensor):
        op = cls._ops[elemwise_op_name(op_name, a.dtype)]
        a, b = a.try_broadcast(b)
        assert a.dtype == b.dtype
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)
        c = Tensor.empty(a.shape, device="cuda", dtype=a.dtype)  # TODO
        assert a.device == "cuda"
        assert b.device == "cuda"
        assert c.device == "cuda"

        a_stride = np.array(a.stride, dtype=np.int32)
        b_stride = np.array(b.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)
        op(
            a.data.ptr, a_stride,  # type: ignore
            b.data.ptr, b_stride,  # type: ignore
            c.data.ptr, c_stride,  # type: ignore
            shape,
            ndim
        )
        return c

    @classmethod
    def _bin_op(cls, op_name, a: Tensor, b: int | float):
        c = Tensor.empty(a.shape, device="cuda", dtype=a.dtype)  # TODO

        assert a.device == "cuda"
        assert c.device == "cuda"

        op = cls._ops[bin_op_name(op_name, a.dtype)]

        ndim = len(a.shape)
        shape = np.array(a.shape, dtype=np.int32)
        a_stride = np.array(a.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)
        op(
            a.data.ptr, a_stride,  # type: ignore
            b,
            c.data.ptr,  c_stride,  # type: ignore
            shape,
            ndim,
        )
        return c

    @classmethod
    def uop(cls, op_name, a: Tensor):
        op = cls._ops[uop_name(op_name, a.dtype)]
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)
        c = Tensor.empty(a.shape, device="cuda", dtype=a.dtype)  # TODO
        assert a.device == "cuda"
        assert c.device == "cuda"

        a_stride = np.array(a.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)
        op(
            a.data.ptr, a_stride,  # type: ignore
            c.data.ptr, c_stride,  # type: ignore
            shape,
            ndim
        )
        return c

    @classmethod
    def bin_op(cls, op: str, a: Tensor, b: Tensor | int | float):
        assert isinstance(a, Tensor)
        if isinstance(b, Tensor):
            return cls.elem_op(op, a, b)
        return cls._bin_op(op, a, b)
