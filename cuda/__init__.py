from __future__ import annotations
import ctypes
import os
import numpy as np
from .alloc import CudaAllocator, Buffer
from .utils import (
    promote_dtype,
    promote_uop_dtype,
    compile_cuda_code,
    get_cuda_code,
)
from .bin import Binary
import grad as grad_ops
from .ops import (
    register_ops,
    elemwise_op_name,
    reduceop_name,
    reduction_op_name,
    uop_name,
    setitem_op_name,
)
from typing import Optional, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from grad import (
        ElemWiseBackwardFn,
        ElemWiseBackwardFnWrapper,
        UnaryOpBackwardFnWrapper,
        UnaryOpBackwardFn,
        ReduceOpBackwardFnWrapper,
        ReduceOpBackwardFn,
    )
    from tensor import Tensor

CUDA_KERNELS = Binary(compile_cuda_code(get_cuda_code(), "cuda_code"))
register_ops(CUDA_KERNELS)


class CUDA_OPS:
    @overload
    @classmethod
    def elem_op(
        cls,
        op_name: str,
        a: Tensor,
        b: Tensor,
        backward_fn: None = None,
        floating_op: bool = False,
        out: Optional[Tensor] = None,
    ) -> Tensor: ...

    @overload
    @classmethod
    def elem_op(
        cls,
        op_name: str,
        a: Tensor,
        b: Tensor,
        backward_fn: ElemWiseBackwardFnWrapper,
        floating_op: bool = False,
        out: Optional[Tensor] = None,
    ) -> tuple[Tensor, ElemWiseBackwardFn]: ...

    @classmethod
    def elem_op(
        cls,
        op_name: str,
        a: Tensor,
        b: Tensor,
        backward_fn: Optional[ElemWiseBackwardFnWrapper] = None,
        floating_op=False,
        out: Optional[Tensor] = None,
    ):
        from tensor import Tensor

        assert a.shape == b.shape
        if out is None:
            if op_name in ["lt", "le", "gt", "ge"]:
                out_dtype = np.dtype(np.bool)
            else:
                out_dtype = np.dtype(promote_dtype(a.dtype, b.dtype, floating_op))
        else:
            out_dtype = out.dtype

        kernel = CUDA_KERNELS.get(
            elemwise_op_name(op_name, a.dtype, b.dtype, out_dtype)
        )

        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)
        if out is not None:
            c = out
            assert a.shape == out.shape
        else:
            c = Tensor.empty(a.shape, dtype=out_dtype)

        a_stride = np.array(a.stride, dtype=np.int32)
        b_stride = np.array(b.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)

        kernel.launch(
            a.data.ptr,
            a_stride,
            b.data.ptr,
            b_stride,
            c.data.ptr,
            c_stride,
            shape,
            ndim,
        )
        if backward_fn is not None:
            return c, backward_fn(a, b, c)
        return c

    @classmethod
    @overload
    def uop(
        cls, op_name: str, a: Tensor, backward_fn: None = None, floating_op: bool = True
    ) -> Tensor: ...

    @classmethod
    @overload
    def uop(
        cls,
        op_name: str,
        a: Tensor,
        backward_fn: UnaryOpBackwardFnWrapper,
        floating_op: bool = True,
    ) -> tuple[Tensor, UnaryOpBackwardFn]: ...

    @classmethod
    def uop(
        cls,
        op_name: str,
        a: Tensor,
        backward_fn: Optional[UnaryOpBackwardFnWrapper] = None,
        floating_op: bool = True,
    ):
        from tensor import Tensor

        out_dtype = promote_uop_dtype(a.dtype, floating_op)
        c = Tensor.empty(a.shape, dtype=out_dtype)
        kernel = CUDA_KERNELS.get(uop_name(op_name, a.dtype, out_dtype))
        shape = np.array(a.shape, dtype=np.int32)
        ndim = len(a.shape)

        a_stride = np.array(a.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)
        kernel.launch(
            a.data.ptr,
            a_stride,
            c.data.ptr,
            c_stride,
            shape,
            ndim,
        )
        if backward_fn is not None:
            return c, backward_fn(a, c)
        return c

    @classmethod
    @overload
    def reduce_op(
        cls,
        op_name,
        a: Tensor,
        axis: int | tuple[int, ...],
        keepdim: bool,
        backward_fn: None = None,
        out_dtype=None,
    ) -> Tensor: ...

    @classmethod
    @overload
    def reduce_op(
        cls,
        op_name,
        a: Tensor,
        axis: int | tuple[int, ...],
        keepdim: bool,
        backward_fn: ReduceOpBackwardFnWrapper,
        out_dtype=None,
    ) -> tuple[Tensor, ReduceOpBackwardFn]: ...

    @classmethod
    def reduce_op(
        cls,
        op_name,
        a: Tensor,
        axis: int | tuple[int, ...],
        keepdim: bool,
        backward_fn: Optional[ReduceOpBackwardFnWrapper] = None,
        out_dtype=None,
    ):
        from tensor import Tensor

        if out_dtype is None:
            out_dtype = a.dtype

        if axis == () and (os.getenv("USE_REDUCTION", "1") != "0"):
            # this is order of 1000s faster
            kernel = CUDA_KERNELS.get(
                reduction_op_name(op_name, str(a.dtype), str(out_dtype))
            )
            out = Tensor.empty((), dtype=out_dtype)
            kernel.launch(
                a.data.ptr,
                out.data.ptr,
                a.size,
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
                    shape.pop(a - i)
                    i += 1
            return shape

        kernel = CUDA_KERNELS.get(reduceop_name(op_name, str(a.dtype), str(out_dtype)))
        c = Tensor.empty(get_shape(list(a.shape), keepdim=False), dtype=out_dtype)

        a_shape = np.array(a.shape, dtype=np.int32)
        c_shape = np.array(c.shape, dtype=np.int32)
        a_stride = np.array(a.stride, dtype=np.int32)
        c_stride = np.array(c.stride, dtype=np.int32)

        kernel.launch(
            a.data.ptr,
            a_stride,
            a_shape,
            c.data.ptr,
            c_stride,
            c_shape,
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
        from tensor import Tensor

        assert a.ndim in [2, 3]
        assert b.ndim in [2, 3]
        assert a.dtype == b.dtype == np.float32

        if b.ndim == 2:
            assert a.shape[-1] == b.shape[0], f"{a.shape} @ {b.shape}"
        else:
            assert a.shape[-1] == b.shape[1], f"{a.shape} @ {b.shape}"

        if a.ndim == 2 and b.ndim == 2:
            # both 2d
            BATCH = 1
            M, K = a.shape
            K, N = b.shape
            out = Tensor.empty((M, N), dtype=np.float32)
            a_stride = [0] + list(a.stride)
            b_stride = [0] + list(b.stride)
            out_stride = [0] + list(out.stride)
        else:
            a_stride = a.stride
            b_stride = b.stride
            if b.ndim == 2:
                # a 3d b 2d
                BATCH, M, K = a.shape
                K, N = b.shape
                b_stride = [0] + list(b.stride)
            elif a.ndim == 2:
                # a 2d b 3d
                M, K = a.shape
                BATCH, K, N = b.shape
                a_stride = [0] + list(a.stride)
            else:
                # both 3d
                assert a.shape[0] == b.shape[0], f"{a.shape} @ {b.shape}"
                BATCH, M, K = a.shape
                BATCH, K, N = b.shape

            out = Tensor.empty((BATCH, M, N), dtype=np.float32)
            out_stride = out.stride

        kernel = CUDA_KERNELS.get("matmul_batched")
        kernel.launch(
            a.data.ptr,
            b.data.ptr,
            out.data.ptr,
            np.array(a_stride, dtype=np.int32),
            np.array(b_stride, dtype=np.int32),
            np.array(out_stride, dtype=np.int32),
            BATCH,
            M,
            K,
            N,
        )
        return out, grad_ops.matmul_backward(a, b)

    @classmethod
    def copy_out(cls, src: Tensor, dst: Tensor):
        kernel_name = f"copy_out_{src.dtype}_{dst.dtype}"
        kernel = CUDA_KERNELS.get(kernel_name)

        src_shape = np.array(src.shape, dtype=np.int32)
        src_stride = np.array(src.stride, dtype=np.int32)
        assert dst.is_contiguous

        kernel.launch(
            src.data.ptr,
            src_shape,
            src_stride,
            dst.data.ptr,
            src.ndim,
        )
        return dst

    @classmethod
    def copy_out_indices(cls, src: Tensor, indices: list[np.typing.NDArray]):
        from tensor import Tensor

        assert False
        # TODO: maybe we need offset ?
        dst_shape = tuple(len(i) if i.ndim != 0 else 1 for i in indices)
        kernel_name = f"copy_out_indices_{src.dtype}"
        kernel = CUDA_KERNELS.get(kernel_name)

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

        indices_arr = IndexArrayType(
            *[
                ctypes.c_void_p(Tensor.from_numpy(arr.astype(np.int32)).data.ptr.value)
                for arr in indices
            ]
        )

        indices_ptr = ctypes.cast(indices_arr, ctypes.POINTER(ctypes.c_void_p))

        kernel.launch(
            src.data.ptr,
            src_shape,
            src_stride,  # type: ignore
            dst.data.ptr,
            indices_ptr,
            dst_shape,  # type: ignore
            src.ndim,
        )
        CudaAllocator.synchronize()
        return dst

    @classmethod
    def copy_to(cls, data: Buffer, dst: Tensor):
        kernel_name = f"copy_out_{dst.dtype}"
        kernel = CUDA_KERNELS.get(kernel_name)

        src_shape = np.array(data.shape, dtype=np.int32)
        src_stride = np.array(data.stride, dtype=np.int32)

        # dst_shape = np.array(dst.shape, dtype=np.int32)
        # dst_stride = np.array(dst.stride, dtype=np.int32)

        kernel.launch(
            data.ptr,
            src_shape,
            src_stride,
            dst.data.ptr,
            dst.ndim,
        )
        return dst

    @classmethod
    def setitem_op(
        cls, t: Tensor, condition: Tensor, value: Tensor | int | float | bool
    ):
        from tensor import Tensor

        if not isinstance(value, Tensor):
            value = Tensor.from_numpy(np.array(value, dtype=t.dtype))
            value = value.expand(*condition.shape)

        assert condition.shape == value.shape
        assert t.dtype == value.dtype
        assert t._broadcastable(condition)
        assert condition.is_contiguous
        assert value.shape == t.shape

        kernel_name = setitem_op_name(t.dtype)
        kernel = CUDA_KERNELS.get(kernel_name)

        value_stride = np.array(value.stride, dtype=np.int32)

        t_shape = np.array(t.shape, dtype=np.int32)
        t_stride = np.array(t.stride, dtype=np.int32)

        kernel.launch(
            value.data.ptr,
            value_stride,
            condition.data.ptr,
            t.data.ptr,
            t_shape,
            t_stride,
            t.ndim,
        )
        if t.requires_grad and grad_ops.Grad.is_on():

            def backward(gradient: Tensor):
                cls.setitem_op(
                    gradient,
                    condition,
                    0,  # TODO: we mught need to get the grad from 'value'
                )
                return gradient

            t._set_backward_fn(grad_ops.InplaceFunction(backward, t))
