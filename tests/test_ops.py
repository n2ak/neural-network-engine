import numpy as np
from numpy.typing import NDArray
from tensor import Tensor


def check(a1: NDArray, a2: NDArray):
    assert a1.shape == a2.shape
    assert a1.strides == a2.strides
    assert np.allclose(a1, a2), np.abs(a1-a2).mean()


def from_numpy(a): return Tensor.from_numpy(a).cuda()


def test_elemwise_ops():
    funcs = [
        lambda a, b:  a + b,
        lambda a, b:  a * b,
        lambda a, b:  a / b,
        lambda a, b:  a - b,
    ]

    shape = (3, 7, 5, 9)
    a = np.random.randn(*shape)
    b = np.random.randn(*shape)

    for func in funcs:
        check(func(a, b), func(from_numpy(a), from_numpy(b)).numpy())


def test_elemwise_ops_broadcast():
    funcs = [
        lambda a, b:  a + b,
        lambda a, b:  a * b,
        lambda a, b:  a / b,
        lambda a, b:  a - b,
    ]

    a = np.random.randn(3, 7, 5, 9)
    b = np.random.randn(7, 1, 9)

    for func in funcs:
        check(func(a, b), func(from_numpy(a), from_numpy(b)).numpy())


def test_bin_ops():
    funcs = [
        lambda a, b:  a + b,
        lambda a, b:  a * b,
        lambda a, b:  a / b,
        lambda a, b:  a - b,
    ]

    shape = (3, 7, 5, 9)
    a = np.random.randn(*shape)
    b = 3
    for func in funcs:
        check(func(a, b), func(from_numpy(a), b).numpy())


def test_uops():
    import torch
    ops = [
        ("exp", lambda x: np.exp(x), lambda x: x.exp()),
        ("log", lambda x: np.log(x), lambda x: x.log()),
        ("log2", lambda x: np.log2(x), lambda x: x.log2()),
        ("expand", lambda x: torch.from_numpy(x).expand(
            *expand_shape).numpy(), lambda x: x.expand(*expand_shape)),
        ("transpose", lambda x: torch.from_numpy(x).transpose(
            *T_shape).numpy(), lambda x: x.transpose(*T_shape)),
        ("permute", lambda x: torch.from_numpy(x).permute(
            *permute_shape).numpy(), lambda x: x.permute(*permute_shape)),
    ]
    shape = (3, 5, 7)
    T_shape = 2, 1
    permute_shape = (1, 2, 0)
    expand_shape = 2, 3, 3, 5, 7
    a = np.random.randn(*shape)+10
    for opname, func1, func2 in ops:
        print(opname)
        check(func1(a), func2(from_numpy(a)).numpy())
