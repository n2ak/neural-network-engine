import numpy as np
import torch as T
from _test_utils import check, from_torch, from_numpy, check_numpy, check_tensor


def test_elemwise_ops():
    for dtype1 in [T.float32, T.float64, T.int32, T.int64]:
        for dtype2 in [T.float32, T.float64, T.int32, T.int64]:
            funcs = [
                ("add", lambda a, b:  a + b, True),
                ("sub", lambda a, b:  a - b, True),
                ("mul", lambda a, b:  a * b, True),
                ("div", lambda a, b:  a / b, True),
                ("lt", lambda a, b:  a < b, False),
                ("le", lambda a, b:  a <= b, False),
                ("gt", lambda a, b:  a > b, False),
                ("ge", lambda a, b:  a >= b, False),
            ]

            shape = (3, 7, 5, 9)
            for name, func, check_grad in funcs:
                check_grad = check_grad and (
                    dtype1.is_floating_point or dtype2.is_floating_point)

                a = T.randn(*shape).to(dtype1)+10
                b = T.randn(*shape).to(dtype2)+10

                print("** Test", name, a.dtype, b.dtype, check_grad)
                check(func, (a, b), check_grad=check_grad)


def test_elemwise_ops_broadcast():
    for dtype1 in [T.float32, T.float64, T.int32, T.int64]:
        for dtype2 in [T.float32, T.float64, T.int32, T.int64]:
            funcs = [
                ("add", lambda a, b:  a + b, True),
                ("sub", lambda a, b:  a - b, True),
                ("mul", lambda a, b:  a * b, True),
                ("div", lambda a, b:  a / b, True),
                ("lt", lambda a, b:  a < b, False),
                ("le", lambda a, b:  a <= b, False),
                ("gt", lambda a, b:  a > b, False),
                ("ge", lambda a, b:  a >= b, False),
            ]

            for name, func, check_grad in funcs:
                a = T.randn(3, 1, 5, 9).to(dtype1)+10
                b = T.randn(7, 1, 9).to(dtype2)+10

                check_grad = check_grad and (
                    dtype1.is_floating_point or dtype2.is_floating_point)

                print("** Test", name, a.dtype, b.dtype, check_grad)

                check(func, (a, b), check_grad=check_grad)


def test_elemwise_ops_inplace():
    for dtype1 in [T.float32, T.float64, T.int32, T.int64]:
        def isub(a, b):
            a -= b
            return a
        funcs = [
            ("isub", isub),
        ]

        for name, func in funcs:
            a = T.randn(3, 1, 5, 9).to(dtype1)+10
            b = T.randn(3, 1, 5, 9).to(dtype1)+10
            print("** Test", name, a.dtype, b.dtype)

            check(func, (a, b), check_grad=False)


def test_bin_ops():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        funcs = [
            ("add", lambda a, b:  a + b),
            ("mul", lambda a, b:  a * b),
            ("div", lambda a, b:  a / b),
            ("sub", lambda a, b:  a - b),
        ]

        shape = (3, 7, 5, 9)
        for name, func in funcs:
            check_grad = dtype.is_floating_point

            a = T.randn(*shape).to(dtype)
            b = 3

            print("** Test", name, dtype, check_grad)
            check(func, (a, b), check_grad=check_grad)


def test_bin_ops_inplace():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        def isub(a, b):
            a -= b
            return a
        funcs = [
            ("isub", isub),
        ]

        shape = (3, 7, 5, 9)
        for name, func in funcs:
            check_grad = dtype.is_floating_point

            a = T.randn(*shape).to(dtype)
            b = 3

            print("** Test", name, dtype, check_grad)
            check(func, (a, b), check_grad=False)


def test_slicing():

    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        shape = (70, 5, 90)

        for i, slices in enumerate([
            (slice(1, 2, None), slice(2, None, 2), slice(2, None, 4)),
            (slice(None, 2, 3), slice(0, 10, 3), slice(2, None, 4)),
            (slice(None, None, 3), slice(0, 10, None), slice(None, None, 4)),
            (slice(100, 3, None), slice(100, None, 2), slice(4, None, 2))
        ]):
            a1 = np.random.randn(*shape).astype(dtype)
            a2 = from_numpy(a1)
            print("** Test", i+1, dtype)
            # contiguous=True tensor.numpy() returns a contiguous
            check_numpy(a1[slices], a2[slices].numpy())


def test_nested_slicing():

    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        shape = (70, 5, 10, 90)

        for i, slices in enumerate([
            (slice(1, 2, None), slice(2, None, 2), slice(2, None, 4)),
            (slice(None, 2, 3), slice(0, 10, 3), slice(2, None, 4)),
            (slice(None, None, 3), slice(0, 10, None), slice(None, None, 4)),
            (slice(100, 3, None), slice(100, None, 2), slice(4, None, 2))
        ]):
            a1 = np.random.randn(*shape).astype(dtype)
            a2 = from_numpy(a1)
            print("** Test", i+1, dtype)
            # contiguous=True tensor.numpy() returns a contiguous
            check_numpy(a1[1:, :3, 2][slices], a2[1:, :3, 2][slices].numpy())


def test_view_offset():

    from tensor import Tensor

    def test(slices1, slices2):
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
    test(
        (slice(1, 2, None), slice(2, None, 2), slice(None, None, 4)),
        (slice(1, 2, None), slice(2, None, 2), slice(4, None, 2))
    )
    test(
        (2, slice(2, None, 2), slice(None, None, 4)),
        (slice(2, None, 2), slice(4, None, 2))
    )
    test(
        (slice(2, None, 2), slice(4, None, 2)),
        (2, slice(2, None, 2), slice(None, None, 4)),
    )


def test_reduce_axis_ops():
    for keepdim in [False, True]:  # TODO: add True
        for dtype in [T.float32, T.float64, T.int32, T.int64]:
            torch_dtype = T.float32
            if dtype == T.float64:
                torch_dtype = T.float64
            ops = [
                ("sum", lambda x: x.sum(sum_axis, keepdim=keepdim), True),
                ("max", lambda x: x.max(max_axis, keepdim=keepdim).values
                    if isinstance(x, T.Tensor) else x.max(max_axis, keepdim=keepdim), False),
                ("mean", lambda x: x.mean(mean_axis, dtype=torch_dtype, keepdim=keepdim)
                    if isinstance(x, T.Tensor) else x.mean(mean_axis, keepdim=keepdim), True),
            ]
            shape = (3, 5, 7, 9)
            mean_axis = 1, 2
            max_axis = 2  # torch only accpets one axis for max
            sum_axis = 1, 2, 3
            for opname, func, check_grad in ops:
                a = T.rand(shape).to(dtype)
                print("** Test", f"{dtype=}, {opname=}, {keepdim=}")
                check(func, (a,), check_grad=check_grad)


def test_reduce_ops_no_axis():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        torch_dtype = T.float32
        if dtype == T.float64:
            torch_dtype = T.float64
        ops = [
            ("sum", lambda x: x.sum(), True),
            ("max", lambda x: x.max(), False),
            ("mean", lambda x: x.mean(dtype=torch_dtype)
                if isinstance(x, T.Tensor) else x.mean(), True),
        ]
        shape = (3, 5, 7, 9)
        for opname, func, check_grad in ops:
            a = (T.randn(*shape) - 100).to(dtype)
            print("** Test", dtype, opname, check_grad)
            check(func, (a,), check_grad=check_grad)


def test_uops():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        ops = [
            ("exp", lambda x: x.exp(), True),
            ("log", lambda x: x.log(), True),
            ("log2", lambda x: x.log2(), True),
        ]
        shape = (3, 5, 7)
        for opname, func, check_grad in ops:
            a = (T.rand(*shape)+1).to(dtype)
            print("** Test", opname, check_grad, dtype)
            check(func, (a,), check_grad=check_grad)


def test_views():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        ops = [
            ("view1", lambda x: x.view(-1), True),
            ("view4", lambda x: x.view(1, -1, 1), True),
            ("view2", lambda x: x.view(*view_shape), True),
            ("view3", lambda x: x.view(*view_shape2), True),
            ("expand", lambda x: x.expand(*expand_shape), True),
            ("transpose", lambda x: x.transpose(*T_shape), True),
            ("permute", lambda x: x.permute(*permute_shape), True),
        ]
        shape = (3, 5, 7)
        T_shape = 2, 1
        permute_shape = (1, 2, 0)
        expand_shape = 2, 3, 3, 5, 7
        view_shape = 5, 7, 3
        view_shape2 = 5, -1
        for opname, func, check_grad in ops:
            a = T.randn(*shape).to(dtype)
            print("** Test", opname)
            check(func, (a,), check_grad=check_grad)


def test_matmul():
    print()

    def func(a, b):
        return a @ b

    for a, b in [
        (T.randn(3, 5, 7), T.randn(7, 9)),
        (T.randn(3, 5, 7), T.randn(3, 7, 9)),
        (T.randn(5, 7), T.randn(3, 7, 9)),
    ]:
        print("** Test", "matmul", a.shape, "@", b.shape)
        check(func, (a, b), check_grad=True)


def test_complex():
    for dtype in [T.float32, T.float64, T.int32, T.int64]:
        a_shape = (3, 5, 7, 9)
        b_shape = (5, 1, 9)

        def func(a, b):
            c = (a*b)+10
            c[c < 1] = 29
            return (c / 12.0).mean()

        a = T.randn(*a_shape).to(dtype)
        b = T.randn(*b_shape).to(dtype)
        check(func, (a, b), check_grad=True)


def test_other_ops():
    def relu(a):
        a = a+0
        a[a < 0] = 0
        return a
    for func in [relu]:
        a = T.randn(3, 5, 7)
        check(func, (a,), check_grad=True)


def test_softmax():
    import nn
    batch = 9
    outc = 10

    input1 = T.randn((batch, outc)).requires_grad_(True)
    input2 = from_torch(input1)

    res1 = T.nn.functional.softmax(input1, -1)
    res2 = nn.softmax(input2, -1)

    check_tensor(
        (input1,),
        (input2,),
        res1,
        res2
    )


def test_log_softmax():
    import nn
    batch = 9
    outc = 10

    tlogits = T.randn((batch, outc)).requires_grad_(True)

    logits = from_torch(tlogits)

    tloss = T.nn.functional.log_softmax(tlogits, -1)
    loss = nn.log_softmax(logits, -1)

    check_tensor(
        (tlogits,),
        (logits,),
        tloss,
        loss
    )


def test_nll():
    import nn
    batch = 9
    outc = 10

    tlogits = T.randn((batch, outc))
    tlogits[tlogits > 0] = 0

    tlogits = tlogits.requires_grad_(True)
    ty = T.randint(0, outc, (batch,))

    logits = from_torch(tlogits)
    y = from_torch(ty)

    tloss = T.nn.functional.nll_loss(tlogits, ty)
    loss = nn.negative_log_likelihood(logits, y)

    check_tensor(
        (tlogits,),
        (logits,),
        tloss,
        loss
    )


def test_cross_entropy():
    import nn
    import grad
    batch = 9
    outc = 10
    tlogits = T.randn((batch, outc)).requires_grad_(True)
    ty = T.randint(0, outc, (batch,), dtype=T.int32)

    logits = from_torch(tlogits)
    y = from_torch(ty)

    tloss = T.nn.functional.cross_entropy(tlogits, ty.long())
    with grad.Grad.on():
        loss = nn.cross_entropy(logits, y)

    check_tensor(
        (tlogits,),
        (logits,),
        tloss,
        loss,
        check_grad=True
    )
