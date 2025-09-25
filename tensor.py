from typing import Self
import numpy as np
from cuda import CudaAllocator, _cuda_ops, bin_op_name, elemwise_op_name, uop_name, Buffer


class Tensor:
    def __init__(self, data: Buffer | np.typing.NDArray, device, shape=None, dtype=None) -> None:
        self.data = data
        self.device = device
        if shape is None:
            assert isinstance(data, np.ndarray)
            shape = data.shape
        if dtype is None:
            assert isinstance(data, np.ndarray)
            dtype = data.dtype
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def from_numpy(data):
        assert isinstance(data, np.ndarray)
        return Tensor(data, device="cpu")

    def cuda(self):
        assert self.device == "cpu"
        data = CudaAllocator.to_cuda(self.data)  # type: ignore
        return Tensor(
            data,
            device="cuda",
            shape=self.shape,
            dtype=self.dtype
        )

    def cpu(self):
        assert self.device == "cuda"
        data = CudaAllocator.from_cuda(
            self.data, self.shape, self.dtype)  # type: ignore
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
    def empty(cls, shape, device="cpu", dtype=None):
        if device == "cuda":
            data = CudaAllocator.alloc_empty(shape=shape, dtype=dtype)
        else:
            data = np.empty(shape, dtype=dtype)
        return Tensor(
            data,
            shape=shape,
            device=device,
            dtype=dtype
        )

    def astype(self, dtype) -> "Tensor":
        if dtype == self.dtype:
            return self
        if self.device == "cpu":
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
        if self.device == "cuda":
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


class CUDA_OPS:
    _ops = _cuda_ops

    @classmethod
    def elem_op(cls, op_name, a: Tensor, b: Tensor):
        op = cls._ops[elemwise_op_name(op_name, a.dtype)]
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)
        c = Tensor.empty(a.shape, device="cuda", dtype=a.dtype)  # TODO
        assert a.device == "cuda"
        assert b.device == "cuda"
        assert c.device == "cuda"
        op(a.data.ptr, b.data.ptr, c.data.ptr, shape, ndim)  # type: ignore
        return c

    @classmethod
    def _bin_op(cls, op_name, a: Tensor, b: int | float):
        op = cls._ops[bin_op_name(op_name, a.dtype)]
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)
        c = Tensor.empty(a.shape, device="cuda", dtype=a.dtype)  # TODO
        assert a.device == "cuda"
        assert c.device == "cuda"
        op(a.data.ptr, b, c.data.ptr, shape, ndim)  # type: ignore
        return c

    @classmethod
    def uop(cls, op_name, a: Tensor):
        op = cls._ops[uop_name(op_name, a.dtype)]
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)
        c = Tensor.empty(a.shape, device="cuda", dtype=a.dtype)  # TODO
        assert a.device == "cuda"
        assert c.device == "cuda"
        op(a.data.ptr, c.data.ptr, shape, ndim)  # type: ignore
        return c

    @classmethod
    def bin_op(cls, op: str, a: Tensor, b: Tensor | int | float):
        assert isinstance(a, Tensor)
        if isinstance(b, Tensor):
            return cls.elem_op(op, a, b)
        return cls._bin_op(op, a, b)
