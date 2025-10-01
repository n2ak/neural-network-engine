from __future__ import annotations
import os
import numpy as np
from typing import Self, Optional, overload, TYPE_CHECKING
from cuda import _cuda_ops
from cuda.alloc import CudaAllocator, Buffer
from cuda.utils import promote_dtype, promote_uop_dtype, assert_cuda_error
from cuda.op_names import *
import grad as grad_ops
from grad import differentiable_function, DifferentiableFunction, broadcastable
if TYPE_CHECKING:
    from grad import ElemWiseBackwardFn, ElemWiseBackwardFnWrapper, UnaryOpBackwardFnWrapper, UnaryOpBackwardFn, ReduceOpBackwardFnWrapper, ReduceOpBackwardFn


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
    def shape(self): return self.data.shape
    @property
    def stride(self): return self.data.stride

    @property
    def stride_bytes(self):
        return tuple(s * self.dtype.itemsize for s in self.data.stride)

    @property
    def dtype(self): return self.data.dtype

    @property
    def is_contiguous(self):
        stride = tuple(self.stride)
        expected_stride = stride_from_shape(self.shape)
        return expected_stride == stride

    @staticmethod
    def from_numpy(data: np.typing.NDArray | int | float):
        if isinstance(data, (int, float)):
            data = np.array(data)
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

    @broadcastable
    @differentiable_function(2)
    def __add__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("add", self, other, backward_fn=grad_ops.add_backward)

    @broadcastable
    @differentiable_function(2)
    def __mul__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("mul", self, other, backward_fn=grad_ops.mul_backward)

    @broadcastable
    @differentiable_function(2)
    def __sub__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("sub", self, other, backward_fn=grad_ops.sub_backward)

    @broadcastable
    @differentiable_function(2)
    def __truediv__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("div", self, other, backward_fn=grad_ops.truediv_backward, floating_op=True)

    # @differentiable_function(2)
    @broadcastable
    def __rtruediv__(self, other: int | float):
        self, tensor_other = self.try_broadcast(other)
        return tensor_other / self

    @broadcastable
    def __lt__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("lt", self, other)

    @broadcastable
    def __le__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("le", self, other)

    @broadcastable
    def __gt__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("gt", self, other)

    @broadcastable
    def __ge__(self, other: Self | int | float):
        return CUDA_OPS.elem_op("ge", self, other)

    @differentiable_function(1)
    def exp(self):
        return CUDA_OPS.uop("exp", self, backward_fn=grad_ops.exp_backward)

    @differentiable_function(1)
    def log(self):
        return CUDA_OPS.uop("log", self, backward_fn=grad_ops.log_backward)

    @differentiable_function(1)
    def log2(self):
        return CUDA_OPS.uop("log2", self, backward_fn=grad_ops.log2_backward)

    def max(self, axis: int | tuple[int, ...] = (), keepdim=False):
        return CUDA_OPS.reduce_op("max", self,  axis=axis, keepdim=keepdim)

    @differentiable_function(1)
    def sum(self, axis: int | tuple[int, ...] = (), keepdim=False):
        out_dtype = self.dtype
        if self.dtype == np.int32:
            out_dtype = np.dtype(np.int64)
        return CUDA_OPS.reduce_op("sum", self, axis=axis, keepdim=keepdim, out_dtype=out_dtype, backward_fn=grad_ops.sum_backward)

    def mean(self, axis: int | tuple[int, ...] = (), keepdim=False):
        d = np.prod([self.shape[a] for a in self._correct_axis(axis)]).item()
        return self.sum(axis=axis, keepdim=keepdim) / d

    def _correct_axis(self, axis: int | tuple[int, ...]):
        axis = as_tuple(axis)
        if axis == ():
            axis = tuple(range(self.ndim))
        for x in axis:
            assert -self.ndim-1 < x < self.ndim
        axis = tuple((self.ndim+x) % self.ndim for x in axis)
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

    @differentiable_function(1)
    def permute(self, *dims: int):
        new_stride = [self.stride[d] for d in dims]
        new_shape = [self.shape[d] for d in dims]
        out = self._as_view(new_shape, new_stride)

        def backward(gradient: Tensor):
            gradient_dims = [dims.index(i) for i in range(self.ndim)]
            gradient = gradient.permute(*gradient_dims)
            return gradient,
        return out, backward

    def contiguous(self):
        if self.is_contiguous:
            return self
        new_stride = stride_from_shape(self.shape)
        # TODO: do it in gpu
        return Tensor.from_numpy(np.ascontiguousarray(self.numpy()))

    def _broadcastable(self, other: "Tensor"):
        for d1, d2 in zip(self.shape[::-1], other.shape[::-1]):
            if d1 == 1 or d2 == 1:
                continue
            if d1 != d2:
                return False
        return True

    def try_broadcast(self, other: Self):
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
        return self.expand(*expected_shape), other.expand(*expected_shape)

    @differentiable_function(1)
    def expand(self, *dims: int):
        if self.shape == dims:
            # this might be a problem
            def backward(gradient: Tensor):
                return gradient,
            return self, backward

        new_shape = dims
        expected_stride = [0] * len(dims)
        expanded_dims = []

        for i, d, sh, s in zip(range(self.ndim), dims[::-1], self.shape[::-1], self.stride[::-1]):
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

            return gradient,

        return self._as_view(new_shape, expected_stride), backward

    @differentiable_function(1)
    def view(self, *dims: int):
        if self.shape != dims:
            assert self.is_contiguous
            dims_l = list(dims)
            m_one = list(filter(lambda x: x == -1, dims))
            match len(m_one):
                case 0: pass
                case 1:
                    index = dims_l.index(-1)
                    dims_l.pop(index)
                    dims_l.insert(
                        index, (self.size // np.prod(dims_l)).astype(int).item())
                    dims = tuple(dims_l)
                case _:
                    raise Exception("Only one dimention can have -1")
            assert np.prod(dims) == self.size
            out = self._as_view(
                dims,
                stride_from_shape(dims)
            )
        else:
            out = self

        def backward(gradient: Tensor):
            return gradient.view(*self.shape),
        return out, backward

    def _as_view(self, shape, stride, ptr_offset=0, slice=None):
        return Tensor(
            data=self.data._as_view(
                shape, stride, offset=ptr_offset, slice=slice),  # type: ignore
        )

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
                _slice = [_slice.start or 0,
                          _slice.stop or dim, _slice.step or 1]

                # [30:...] and the dim is lower than 30 => we clip
                _slice[1] = clip(_slice[1], 0, dim)

                new_dim = math.ceil((_slice[1] - _slice[0]) / _slice[2])
                # clip negative dimensions
                new_dim = clip(new_dim, 0, dim)

                _slice[0] = clip(_slice[0], 0, np.array(dim))
                # _slice[0] = np.where(_slice[0] < shape, _slice[0], 0)

                offset = (_slice[0] * strd)
                # strides at 0 dimensions shouldnt contribute to add offset
                # it doesn't matter because the size is 0, just to match numpy's results
                if new_dim == 0:
                    offset = 0

                new_strd = (_slice[2] * strd)
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
            slice=keys
        )

    def __setitem__(self, keys, value: Self | int | float | bool):
        if isinstance(keys, Tensor):
            if not self._broadcastable(keys):
                raise Exception("Tensor is not broadcastable with the keys")
            CUDA_OPS.setitem_op(self, keys, value)
        else:
            raise NotImplementedError()

    def backward(self, grad=None):
        assert self._requires_gradient
        if grad is None:
            grad = Tensor.from_numpy(np.array(1, dtype=np.float32))

        assert self.shape == grad.shape, (self.shape, grad.shape)

        if (fn := getattr(self, "_backward", None)) is not None:
            # assert isinstance(fn, DifferentiableFunction), type(fn)
            fn.backward(grad)
        else:
            self.grad = grad

    def _set_backward_fn(self, func: DifferentiableFunction):
        self._backward = func

    @property
    def grad(self):
        assert self._requires_gradient, "This tensor doesn't require gradient"
        return self._grad

    @grad.setter
    def grad(self, val: "Tensor"):
        assert self._requires_gradient, "This tensor doesn't require gradient"
        assert not val._requires_gradient, "Gradient tensors should not require gradient"
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
    def requires_grad(self): return self._requires_gradient

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


class CUDA_OPS:
    _kernels = _cuda_ops

    @overload
    @classmethod
    def elem_op(
        cls, op_name: str, a: Tensor, b: Tensor,
        floating_op: bool = False,
    ) -> Tensor: ...

    @overload
    @classmethod
    def elem_op(
        cls, op_name: str, a: Tensor, b: Tensor,
        backward_fn: ElemWiseBackwardFnWrapper,
        floating_op: bool = False,
    ) -> tuple[Tensor, ElemWiseBackwardFn]: ...

    @classmethod
    def elem_op(
        cls, op_name: str, a: Tensor, b: Tensor,
        backward_fn: Optional[ElemWiseBackwardFnWrapper] = None,
        floating_op=False,
    ):
        assert a.shape == b.shape
        if op_name in ["lt", "le", "gt", "ge"]:
            out_dtype = np.dtype(np.bool)
        else:
            out_dtype = np.dtype(promote_dtype(a.dtype, b.dtype, floating_op))

        kernel = cls._kernels[elemwise_op_name(
            op_name, a.dtype, b.dtype, out_dtype)]

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
        if backward_fn is not None:
            return c, backward_fn(a, b, c)
        return c

    @classmethod
    @overload
    def uop(
        cls, op_name: str, a: Tensor,
        floating_op: bool = True
    ) -> Tensor: ...

    @classmethod
    @overload
    def uop(
        cls, op_name: str, a: Tensor,
        backward_fn: UnaryOpBackwardFnWrapper,
        floating_op: bool = True
    ) -> tuple[Tensor, UnaryOpBackwardFn]: ...

    @classmethod
    def uop(
        cls, op_name: str, a: Tensor,
        backward_fn: Optional[UnaryOpBackwardFnWrapper] = None,
        floating_op: bool = True
    ):
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
        if backward_fn is not None:
            return c, backward_fn(a, c)
        return c

    @classmethod
    @overload
    def reduce_op(
        cls, op_name, a: Tensor, axis: int | tuple[int, ...], keepdim: bool,
        out_dtype=None,
    ) -> Tensor: ...

    @classmethod
    @overload
    def reduce_op(
        cls, op_name, a: Tensor, axis: int | tuple[int, ...], keepdim: bool,
        backward_fn: ReduceOpBackwardFnWrapper,
        out_dtype=None,
    ) -> tuple[Tensor, ReduceOpBackwardFn]: ...

    @classmethod
    def reduce_op(
        cls, op_name, a: Tensor, axis: int | tuple[int, ...], keepdim: bool, out_dtype=None,
        backward_fn: Optional[ReduceOpBackwardFnWrapper] = None,
    ):
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
            if backward_fn is not None:
                return out, backward_fn(a, out, (), keepdim)
            return out
        axis = a._correct_axis(axis)

        def get_shape(shape: list[int], keepdim):
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
        c = Tensor.empty(
            get_shape(list(a.shape), keepdim=False), dtype=out_dtype)

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
        )
        c = c.view(*get_shape(list(a.shape), keepdim=keepdim))
        if backward_fn is not None:
            return c, backward_fn(a, c, axis, keepdim)
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
    def copy_out_indices(cls, src: Tensor, indices: list[np.typing.NDArray]):
        assert False
        # TODO: maybe we need offset ?
        dst_shape = tuple(len(i) if i.ndim != 0 else 1 for i in indices)
        kernel_name = f"copy_out_indices_{src.dtype}"
        kernel = cls._kernels[kernel_name]

        dst = Tensor.empty(dst_shape, dtype=src.dtype)

        src_shape = np.array(src.shape, dtype=np.int32)
        src_stride = np.array(src.stride, dtype=np.int32)

        dst_shape = np.array(dst.shape, dtype=np.int32)

        # indices_arr = np.array([
        #     Tensor.from_numpy(arr.astype(np.float32)).data.ptr.value
        #     for arr in indices
        # ], dtype=np.float64)
        import ctypes
        IndexArrayType = ctypes.c_void_p * src.ndim

        indices_arr = IndexArrayType(*[
            ctypes.c_void_p(Tensor.from_numpy(
                arr.astype(np.int32)).data.ptr.value)
            for arr in indices
        ])

        indices_ptr = ctypes.cast(indices_arr, ctypes.POINTER(ctypes.c_void_p))

        kernel(
            src.data.ptr, src_shape, src_stride,  # type: ignore
            dst.data.ptr, indices_ptr, dst_shape,  # type: ignore
            src.ndim,
        )
        CudaAllocator.synchronize()
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

    @classmethod
    def setitem_op(cls, t: Tensor, condition: Tensor, value: Tensor | int | float | bool):
        if not isinstance(value, Tensor):
            value = Tensor.from_numpy(np.array(value, dtype=t.dtype))
            value = value.expand(*condition.shape)

        assert condition.shape == value.shape
        assert t.dtype == value.dtype
        assert t._broadcastable(condition)
        assert condition.is_contiguous
        assert value.shape == t.shape

        kernel_name = setitem_op_name(t.dtype)
        kernel = cls._kernels[kernel_name]

        value_stride = np.array(value.stride, dtype=np.int32)

        t_shape = np.array(t.shape, dtype=np.int32)
        t_stride = np.array(t.stride, dtype=np.int32)

        kernel(
            value.data.ptr, value_stride,  # type: ignore
            condition.data.ptr,   # type: ignore
            t.data.ptr, t_shape, t_stride,
            t.ndim
        )
