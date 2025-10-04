from __future__ import annotations
import numpy as np
from cuda import CUDA_OPS
from cuda.alloc import CudaAllocator, Buffer
import grad as grad_ops
from grad import BackwardFn, differentiable, broadcast

from typing import Self, Optional


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
        self._requires_gradient = False
        self._grad: Optional[Tensor] = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def stride(self):
        return self.data.stride

    @property
    def stride_bytes(self):
        return tuple(s * self.dtype.itemsize for s in self.data.stride)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_contiguous(self):
        stride = tuple(self.stride)
        expected_stride = stride_from_shape(self.shape)
        return expected_stride == stride

    @staticmethod
    def from_numpy(data: np.typing.NDArray | int | float | list):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        assert isinstance(data, np.ndarray), type(data)
        return Tensor(
            CudaAllocator.to_cuda(data),
        )

    @staticmethod
    def randn(*shape: int, dtype=np.float32):
        return Tensor.from_numpy(np.random.randn(*shape).astype(dtype))

    def copy_to(self, out: "Tensor") -> "Tensor":
        assert self.shape == out.shape, (self.shape, out.shape)
        return CUDA_OPS.copy_out(self, out)

    @classmethod
    def empty(
        cls,
        shape,
        dtype: np.typing.DTypeLike = np.float32,
    ):
        data = CudaAllocator.alloc_empty(
            shape=shape, stride=stride_from_shape(shape), dtype=dtype
        )
        stride = stride_from_shape(shape)
        return Tensor(
            data,
        )

    @classmethod
    def zeros(cls, shape, dtype=np.float32):
        return Tensor.from_numpy(np.zeros(shape=shape, dtype=dtype))

    @classmethod
    def ones_like(cls, t: "Tensor"):
        return Tensor.ones(*t.shape, dtype=t.dtype)

    @classmethod
    def ones(cls, *shape: int, dtype: np.typing.DTypeLike = np.float32):
        return Tensor.from_numpy(np.ones(shape=shape, dtype=dtype))

    @classmethod
    def randint(
        cls,
        low: int,
        high: int,
        shape: tuple[int, ...],
        dtype: np.typing.DTypeLike = np.int32,
    ):
        return Tensor.from_numpy(np.random.randint(low, high, size=shape).astype(dtype))

    def astype(self, dtype) -> "Tensor":
        if dtype == self.dtype:
            return self
        return self.copy_to(Tensor.empty(self.shape, dtype))

    def numpy(self) -> np.typing.NDArray:
        data = self.data.numpy()
        assert self.shape == data.shape, (self.shape, data.shape)
        return data

    def __del__(self):
        CudaAllocator.free(self.data)

    @broadcast()
    @differentiable(2)
    def __add__(self, other: Tensor):
        return CUDA_OPS.elem_op("add", self, other, backward_fn=grad_ops.add_backward)

    @broadcast()
    @differentiable(2)
    def __mul__(self, other: Tensor):
        return CUDA_OPS.elem_op("mul", self, other, backward_fn=grad_ops.mul_backward)

    @broadcast()
    @differentiable(2)
    def __sub__(self, other: Tensor):
        return CUDA_OPS.elem_op("sub", self, other, backward_fn=grad_ops.sub_backward)

    @broadcast()
    @differentiable(2)
    def __truediv__(self, other: Tensor):
        return CUDA_OPS.elem_op(
            "div", self, other, backward_fn=grad_ops.truediv_backward, floating_op=True
        )

    # @differentiable_function(2)
    @broadcast()
    def __rtruediv__(self, other: Tensor):
        return other / self

    @broadcast(second_only=True)
    def __iadd__(self, other: Tensor):
        assert self.shape == other.shape
        CUDA_OPS.elem_op("add", self, other.astype(self.dtype), out=self)
        return self

    @broadcast(second_only=True)
    def __isub__(self, other: Tensor):
        assert self.shape == other.shape
        CUDA_OPS.elem_op("sub", self, other.astype(self.dtype), out=self)
        return self

    @broadcast(second_only=True)
    def __imul__(self, other: Tensor):
        assert self.shape == other.shape
        CUDA_OPS.elem_op("mul", self, other.astype(self.dtype), out=self)
        return self

    @broadcast(second_only=True)
    def __idiv__(self, other: Tensor):
        assert self.shape == other.shape
        CUDA_OPS.elem_op("div", self, other.astype(self.dtype), out=self)
        return self

    @broadcast()
    def __lt__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("lt", self, other)  # type: ignore

    @broadcast()
    def __le__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("le", self, other)  # type: ignore

    @broadcast()
    def __gt__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("gt", self, other)  # type: ignore

    @broadcast()
    def __ge__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("ge", self, other)  # type: ignore

    def __pow__(self, other: int):
        assert other == 2
        return self * self

    @differentiable(1)
    def exp(self):
        return CUDA_OPS.uop("exp", self, backward_fn=grad_ops.exp_backward)

    def sqrt(self):
        # TODO add it to uops
        return Tensor.from_numpy(np.sqrt(self.numpy()))

    @differentiable(1)
    def log(self):
        return CUDA_OPS.uop("log", self, backward_fn=grad_ops.log_backward)

    @differentiable(1)
    def log2(self):
        return CUDA_OPS.uop("log2", self, backward_fn=grad_ops.log2_backward)

    def max(self, axis: int | tuple[int, ...] = (), keepdim=False):
        return CUDA_OPS.reduce_op("max", self, axis=axis, keepdim=keepdim)

    @differentiable(1)
    def sum(self, axis: int | tuple[int, ...] = (), keepdim=False):
        out_dtype = self.dtype
        if self.dtype == np.int32:
            out_dtype = np.dtype(np.int64)
        return CUDA_OPS.reduce_op(
            "sum",
            self,
            axis=axis,
            keepdim=keepdim,
            out_dtype=out_dtype,
            backward_fn=grad_ops.sum_backward,
        )

    def mean(self, axis: int | tuple[int, ...] = (), keepdim=False):
        d = np.prod([self.shape[a] for a in self._correct_axis(axis)]).item()
        return self.sum(axis=axis, keepdim=keepdim) / d

    def _correct_axis(self, axis: int | tuple[int, ...]):
        axis = as_tuple(axis)
        if axis == ():
            axis = tuple(range(self.ndim))
        for x in axis:
            assert -self.ndim - 1 < x < self.ndim
        axis = tuple((self.ndim + x) % self.ndim for x in axis)
        return axis

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape, dtype=np.int32).item()

    def transpose(self, dim1: int, dim2: int):
        dims = list(range(self.ndim))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(*dims)

    @differentiable(1)
    def permute(self, *dims: int):
        # TODO: add support for negative dims
        new_stride = [self.stride[d] for d in dims]
        new_shape = [self.shape[d] for d in dims]
        out = self._as_view(new_shape, new_stride)

        def backward(gradient: Tensor):
            gradient_dims = [dims.index(i) for i in range(self.ndim)]
            gradient = gradient.permute(*gradient_dims)
            return (gradient,)

        return out, backward

    def contiguous(self):
        if self.is_contiguous:
            return self
        return self.copy_to(Tensor.empty(self.shape, dtype=self.dtype))

    def _broadcastable(self, other: "Tensor"):
        for d1, d2 in zip(self.shape[::-1], other.shape[::-1]):
            if d1 == 1 or d2 == 1:
                continue
            if d1 != d2:
                return False
        return True

    def try_broadcast(self, other: "Tensor", second_only=False):
        assert isinstance(self, Tensor)
        if isinstance(other, (int, float)):
            other = Tensor.from_numpy(np.array(other, dtype=self.dtype))
            other = other.expand(*self.shape)
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
        if second_only:
            assert self.shape == expected_shape
            return self, other.expand(*expected_shape)
        return self.expand(*expected_shape), other.expand(*expected_shape)

    @differentiable(1)
    def expand(self, *dims: int):
        if self.shape == dims:
            # this might be a problem
            def backward(gradient: Tensor):
                return (gradient,)

            return self, backward

        new_shape = dims
        expected_stride = [0] * len(dims)
        expanded_dims = []

        for i, d, sh, s in zip(
            range(self.ndim), dims[::-1], self.shape[::-1], self.stride[::-1]
        ):
            if sh == d:
                expected_stride[i] = s
            else:
                if sh != 1:
                    raise Exception(f"dimension at {i} should be 1")
                expected_stride[i] = 0
                expanded_dims.append(len(dims) - i - 1)

        expected_stride = expected_stride[::-1]
        expanded_dims = expanded_dims[::-1]

        def backward(gradient: Tensor):
            diff = len(dims) - self.ndim
            if expanded_dims:
                # reduce expanded dims to 1
                gradient = gradient.sum(tuple(expanded_dims), keepdim=True)
            if diff:
                # remove first added dims
                gradient = gradient.sum(tuple(range(diff)))

            return (gradient,)

        return self._as_view(new_shape, expected_stride), backward

    @differentiable(1)
    def view(self, *dims: int):
        if self.shape != dims:
            assert self.is_contiguous
            dims_l = list(dims)
            m_one = list(filter(lambda x: x == -1, dims))
            match len(m_one):
                case 0:
                    pass
                case 1:
                    index = dims_l.index(-1)
                    dims_l.pop(index)
                    dims_l.insert(
                        index, (self.size // np.prod(dims_l)).astype(int).item()
                    )
                    dims = tuple(dims_l)
                case _:
                    raise Exception("Only one dimention can have -1")
            assert np.prod(dims) == self.size
            out = self._as_view(dims, stride_from_shape(dims))
        else:
            out = self

        def backward(gradient: Tensor):
            return (gradient.view(*self.shape),)

        return out, backward

    def _as_view(self, shape, stride, ptr_offset=0, slice=None):
        return Tensor(
            data=self.data._as_view(shape, stride, offset=ptr_offset, slice=slice),
        )

    @differentiable(2)
    def __matmul__(self, other: Self):
        return CUDA_OPS.matmul(self, other)

    def __getitem__(self, keys):
        import math

        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) < self.ndim:
            left = self.ndim - len(keys)
            keys = tuple(list(keys) + [slice(None) for _ in range(left)])
        assert len(keys) == self.ndim

        if any(map(lambda k: isinstance(k, list), keys)):
            # we need to copy
            def to_list(i, k):
                if isinstance(k, (list, tuple)):
                    return np.array(k)
                elif isinstance(k, int):
                    return np.array([k])
                elif isinstance(k, slice):
                    return np.arange(k.start or 0, k.stop or self.shape[i], k.step or 1)
                assert False, type(k)

            keys = [to_list(i, k) for i, k in enumerate(keys)]
            return CUDA_OPS.copy_out_indices(self, keys)

        new_shape: list[int] = []
        new_stride: list[int] = []
        offsets: list[int] = []
        for key, dim, strd in zip(keys, self.shape, self.stride):
            if isinstance(key, slice):
                _slice = key
                _slice = [_slice.start or 0, _slice.stop or dim, _slice.step or 1]

                # [30:...] and the dim is lower than 30 => we clip
                _slice[1] = clip(_slice[1], 0, dim)

                new_dim = math.ceil((_slice[1] - _slice[0]) / _slice[2])
                # clip negative dimensions
                new_dim = clip(new_dim, 0, dim)

                _slice[0] = clip(_slice[0], 0, np.array(dim))
                # _slice[0] = np.where(_slice[0] < shape, _slice[0], 0)

                offset = _slice[0] * strd
                # strides at 0 dimensions shouldnt contribute to add offset
                # it doesn't matter because the size is 0, just to match numpy's results
                if new_dim == 0:
                    offset = 0

                new_strd = _slice[2] * strd
                # strides at 0 dimensions shouldnt add contribute to stride
                # it doesn't matter because the size is 0, just to match numpy's results
                if new_dim == 0:
                    new_strd = strd

                offsets.append(offset)
                new_shape.append(new_dim)
                new_stride.append(new_strd)
            elif isinstance(key, int):
                assert key < dim
                offsets.append(key * strd)
            else:
                assert False, type(key)
        # print(self.shape, "shape->", new_shape)
        # print(self.stride, "stride->", new_stride)
        # print("offsets", offsets)
        return self._as_view(
            shape=tuple(new_shape),
            stride=tuple(new_stride),
            ptr_offset=sum(offsets),
            slice=keys,
        )

    def __setitem__(self, keys, value: Self | int | float | bool):
        if isinstance(keys, Tensor):
            if not self._broadcastable(keys):
                raise Exception("Tensor is not broadcastable with the keys")
            CUDA_OPS.setitem_op(self, keys, value)
        else:
            raise NotImplementedError()

    def backward(self, grad: Optional["Tensor"] = None):
        assert self._requires_gradient, "Tensor doesn't require gradient."
        if grad is None:
            grad = Tensor.from_numpy(np.array(1, dtype=np.float32))

        assert self.shape == grad.shape, (self.shape, grad.shape)

        if (fn := getattr(self, "_backward", None)) is not None:
            # assert isinstance(fn, DifferentiableFunction), type(fn)
            assert isinstance(fn, list)
            for bfn in reversed(fn):
                assert isinstance(bfn, (BackwardFn, grad_ops.InplaceBackwardFn)), type(
                    bfn
                )
                grad = bfn.backward(grad)
        else:
            self.grad = grad

    def _set_backward_fn(self, func: BackwardFn | grad_ops.InplaceBackwardFn):
        if not hasattr(self, "_backward"):
            self._backward = []
        self._backward.append(func)

    @property
    def grad(self):
        assert self._requires_gradient, "This tensor doesn't require gradient"
        return self._grad

    @grad.setter
    def grad(self, val: Optional["Tensor"]):
        if val is None:
            self._grad = None
            return
        assert self._requires_gradient, "This tensor doesn't require gradient"
        assert (
            not val._requires_gradient
        ), "Gradient tensors should not require gradient"
        assert val.shape == self.shape, (
            "The gradient should have the same shape as the tensor, "
            f"Expected: {self.shape}, found {val.shape}"
        )
        if self._grad is None:
            self._grad = val
        else:
            self._grad += val
        # print("Accumulated grad", self)

    @property
    def requires_grad(self):
        return self._requires_gradient

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._requires_gradient = val

    def requires_grad_(self, val: bool):
        if val and self.dtype not in [np.float32, np.float64]:
            raise Exception("Only floating tensors can require grad")
        self._requires_gradient = val
        return self


def clip(x, _min, _max):
    return max(_min, min(x, _max))
