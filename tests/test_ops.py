import torch as T
from _utils import check, from_torch, from_numpy


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


def test_slicing():
    import numpy as np
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        np.random.seed(0)
        shape = (70, 5, 90)
        a = np.random.randn(*shape).astype(dtype)

        for i, slices in enumerate([
            (slice(1, 2, None), slice(2, None, 2), slice(2, None, 4)),
            (slice(None, 2, 3), slice(0, 10, 3), slice(2, None, 4)),
            (slice(None, None, 3), slice(0, 10, None), slice(None, None, 4)),
            (slice(100, 3, None), slice(100, None, 2), slice(4, None, 2))
        ]):
            print(i+1, dtype)
            # contiguous=True tensor.numpy() returns a contiguous
            check(a[slices], from_numpy(a)[slices])


def test_nested_slicing():
    import numpy as np
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        np.random.seed(0)
        T.manual_seed(0)
        shape = (70, 5, 90)
        a = np.random.randn(*shape).astype(dtype)

        for i, slices in enumerate([
            (slice(1, 2, None), slice(2, None, 2), slice(2, None, 4)),
            (slice(None, 2, 3), slice(0, 10, 3), slice(2, None, 4)),
            (slice(None, None, 3), slice(0, 10, None), slice(None, None, 4)),
            (slice(100, 3, None), slice(100, None, 2), slice(4, None, 2))
        ]):
            print(i+1, dtype)
            # contiguous=True tensor.numpy() returns a contiguous
            check(a[1:, :3][slices], from_numpy(a)[
                  1:, :3][slices])


def test_view_offset():
    import numpy as np
    from tensor import Tensor
    slices1 = slice(1, 2, None), slice(2, None, 2), slice(None, None, 4)
    slices2 = slice(1, 2, None), slice(2, None, 2), slice(4, None, 2)

    shape = (70, 5, 90)
    a1 = np.random.randn(*shape)
    t1 = Tensor.from_numpy(a1)
    itemsize = a1.itemsize

    def np_ptr(a):
        return a.__array_interface__["data"][0]//itemsize

    a2 = a1[slices1]
    a3 = a2[slices2]
    a1_ptr = np_ptr(a1)
    a2_ptr = np_ptr(a2)
    a3_ptr = np_ptr(a3)

    t2 = t1[slices1]
    t3 = t2[slices2]
    t1_ptr = t1.data.ptr.value//itemsize  # type: ignore
    t2_ptr = t2.data.ptr.value//itemsize  # type: ignore
    t3_ptr = t3.data.ptr.value//itemsize  # type: ignore

    print(a3_ptr - a2_ptr, a2_ptr - a1_ptr, a3_ptr-a1_ptr)
    print(t3_ptr - t2_ptr, t2_ptr - t1_ptr, t3_ptr-t1_ptr)

    assert a3_ptr - a2_ptr == t3_ptr - t2_ptr
    assert a2_ptr - a1_ptr == t2_ptr - t1_ptr
    assert a3_ptr - a1_ptr == t3_ptr - t1_ptr


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
