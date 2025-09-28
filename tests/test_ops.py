import torch as T
from _utils import check, from_torch


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


def test_reduce_axis_ops():
    for keepdim in [False]:  # TODO: add True
        for dtype in [T.float32, T.float64, T.int32, T.int64]:
            torch_dtype = T.float32
            if dtype == T.float64:
                torch_dtype = T.float64
            ops = [
                ("sum", lambda x: x.sum(sum_axis, keepdim=keepdim)),
                ("max", lambda x: x.max(max_axis, keepdim=keepdim).values
                    if isinstance(x, T.Tensor) else x.max(max_axis, keepdim=keepdim)),
                ("mean", lambda x: x.mean(mean_axis, dtype=torch_dtype, keepdim=keepdim)
                    if isinstance(x, T.Tensor) else x.mean(mean_axis, keepdim=keepdim)),
            ]
            shape = (3, 5, 7, 9)
            mean_axis = 1, 2
            max_axis = 2  # torch only accpets one axis for max
            sum_axis = 1, 2, 3
            a = (T.randn(*shape) + 10).to(dtype)
            for opname, func in ops:
                print(dtype, opname, keepdim)
                check(func(a), func(from_torch(a)))


def test_reduce_ops():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        torch_dtype = T.float32
        if dtype == T.float64:
            torch_dtype = T.float64
        ops = [
            ("sum", lambda x: x.sum()),
            ("max", lambda x: x.max()),
            ("mean", lambda x: x.mean(dtype=torch_dtype)
                if isinstance(x, T.Tensor) else x.mean()),
        ]
        shape = (3, 5, 7, 9)
        a = (T.randn(*shape) - 100).to(dtype)
        for opname, func in ops:
            print(dtype, opname)
            check(func(a), func(from_torch(a)))


def test_reduce_ops_no_axis():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        torch_dtype = T.float32
        if dtype == T.float64:
            torch_dtype = T.float64
        ops = [
            ("sum", lambda x: x.sum()),
            ("max", lambda x: x.max()
                if isinstance(x, T.Tensor) else x.max()),
            ("mean", lambda x: x.mean(dtype=torch_dtype)
                if isinstance(x, T.Tensor) else x.mean()),
        ]
        shape = (3, 5, 7, 9)
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


def test_matmul():
    T.manual_seed(0)
    a = T.randn(3, 5, 7)
    b = T.randn(7, 9)

    def func(a, b):
        return a @ b

    check(func(a, b), func(from_torch(a), from_torch(b)))


def test_complex():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        a_shape = (3, 5, 7, 9)
        b_shape = (5, 1, 9)
        a = (T.randn(*a_shape)+10).to(dtype)
        b = (T.randn(*b_shape)+10).to(dtype)

        def func(a, b):
            return (((a*b)+10) / 12.0).mean()

        check(func(a, b), func(from_torch(a), from_torch(b)))
