import numpy as np
from numpy.typing import NDArray
from tensor import Tensor
import torch as T


def check(a1: T.Tensor, a2: Tensor):
    assert isinstance(a1, T.Tensor)
    assert isinstance(a2, Tensor)

    a1_numpy = a1.numpy()
    a2_numpy = a2.numpy()
    assert a1_numpy.shape == a2_numpy.shape
    assert a1_numpy.dtype == a2_numpy.dtype
    assert a1_numpy.strides == a2_numpy.strides
    assert np.allclose(a1_numpy, a2_numpy), (a1_numpy.flatten()[
        :10], a2_numpy.flatten()[:10])


def from_torch(a: T.Tensor): return Tensor.from_numpy(a.numpy()).cuda()


def test_elemwise_ops():
    for dtype1 in [T.float32, T.float64, T.int32, T.int64]:
        for dtype2 in [T.float32, T.float64, T.int32, T.int64]:
            funcs = [
                ("add", lambda a, b:  a + b),
                ("sub", lambda a, b:  a - b),
                ("mul", lambda a, b:  a * b),
                ("div", lambda a, b:  a / b),
            ]

            shape = (3, 7, 5, 9)
            a = T.randn(*shape).to(dtype1)
            b = T.randn(*shape).to(dtype2)+10

            for name, func in funcs:
                print(name, a.dtype, b.dtype)
                check(func(a, b), func(from_torch(a), from_torch(b)))


def test_elemwise_ops_broadcast():
    for dtype1 in [T.float32, T.float64, T.int32, T.int64]:
        for dtype2 in [T.float32, T.float64, T.int32, T.int64]:
            funcs = [
                ("add", lambda a, b:  a + b),
                ("sub", lambda a, b:  a - b),
                ("mul", lambda a, b:  a * b),
                ("div", lambda a, b:  a / b),
            ]

            a = T.randn(3, 1, 5, 9).to(dtype1)
            b = T.randn(7, 1, 9).to(dtype2)+10

            for name, func in funcs:
                print(name, a.dtype, b.dtype)
                check(func(a, b), func(from_torch(a), from_torch(b)))


def test_bin_ops():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        funcs = [
            ("add", lambda a, b:  a + b),
            ("mul", lambda a, b:  a * b),
            ("div", lambda a, b:  a / b),
            ("sub", lambda a, b:  a - b),
        ]

        shape = (3, 7, 5, 9)
        a = T.randn(*shape).to(dtype)
        b = 3
        for name, func in funcs:
            print(name, a.dtype, b)
            check(func(a, b), func(from_torch(a), b))


def test_reduce_ops():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        torch_dtype = T.float32
        if dtype == T.float64:
            torch_dtype = T.float64
        ops = [
            ("sum", lambda x: x.sum(sum_axis)),
            ("max", lambda x: x.max(max_axis).values
                if isinstance(x, T.Tensor) else x.max(max_axis)),
            ("mean", lambda x: x.mean(mean_axis, dtype=torch_dtype)
                if isinstance(x, T.Tensor) else x.mean(mean_axis)),
        ]
        shape = (3, 5, 7, 9)
        mean_axis = 1, 2
        max_axis = 2  # torch only accpets one axis for max
        sum_axis = 1, 2, 3
        a = (T.randn(*shape) + 10).to(dtype)
        for opname, func in ops:
            print(dtype, opname)
            check(func(a), func(from_torch(a)))


def test_uops():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        ops = [
            ("exp", lambda x: x.exp()),
            ("log", lambda x: x.log()),
            ("log2", lambda x: x.log2()),
            ("expand", lambda x: x.expand(*expand_shape)),
            ("transpose", lambda x: x.transpose(*T_shape)),
            ("permute", lambda x: x.permute(*permute_shape)),
        ]
        shape = (3, 5, 7)
        T_shape = 2, 1
        permute_shape = (1, 2, 0)
        expand_shape = 2, 3, 3, 5, 7
        a = (T.randn(*shape)+10).to(dtype)
        for opname, func in ops:
            print(opname)
            check(func(a), func(from_torch(a)))
