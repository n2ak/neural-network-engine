import torch
import numpy as np
from tensor import Tensor
from typing import Callable, Any, TypeVar, Unpack, TypeVarTuple, ParamSpecArgs, ParamSpec
from grad import Grad
T = TypeVar("T")
Ts = TypeVarTuple("Ts")
P = ParamSpec("P")
A = ParamSpecArgs(P)


def check(
    func: Callable[..., Tensor | torch.Tensor],
    inputs_torch: tuple[torch.Tensor | int, ...],
    rtol=1e-05,
    atol=1e-08,
    check_grad=False,
):

    if check_grad:
        inputs_torch = tuple(i.requires_grad_(i.dtype.is_floating_point) if isinstance(
            i, torch.Tensor) else i for i in inputs_torch)
        check_grad = any(map(lambda t: t.requires_grad if isinstance(
            t, torch.Tensor) else False, inputs_torch))

    inputs_tensors = tuple(from_torch(i) if isinstance(
        i, torch.Tensor) else i for i in inputs_torch)

    a1: torch.Tensor = func(*inputs_torch)  # type: ignore
    with Grad.on(check_grad):
        a2: Tensor = func(*inputs_tensors)  # type: ignore

    print(
        "Input Dtypes:",
        [inp.dtype if isinstance(inp, Tensor)else type(inp)
         for inp in inputs_tensors],
        "Result type:", a2.dtype
    )
    if check_grad:
        a1.backward(torch.ones_like(a1))
        a2.backward(Tensor.from_numpy(np.ones(a2.shape, dtype=a2.dtype)))

    assert isinstance(a1, (torch.Tensor, np.ndarray)), type(a1)
    assert isinstance(a2, Tensor), type(a2)

    # assert a1.stride() == a2.stride, (a1.stride(), a2.stride)

    a1_numpy = a1.numpy(force=True) if isinstance(a1, torch.Tensor)else a1
    a2_numpy = a2.numpy()

    check_result(a1_numpy, a2_numpy, rtol=rtol, atol=atol)

    if check_grad:
        for i, t1, t2 in zip(range(len(inputs_torch)), inputs_torch, inputs_tensors):
            if not isinstance(t1, torch.Tensor) or not t1.requires_grad:
                continue
            assert isinstance(t2, Tensor)

            assert t1.requires_grad
            assert t2.requires_grad

            print(f"Checking tensor {i+1}", t2)
            check_result(t1.grad.numpy(), t2.grad.numpy(), check_dtype=False)


def check_result(a1_numpy: np.ndarray, a2_numpy: np.ndarray, rtol=1e-05,
                 atol=1e-08, check_dtype=True):
    # if contiguous:
    #     a1_numpy = np.ascontiguousarray(a1_numpy)
    #     a2_numpy = np.ascontiguousarray(a2_numpy)
    if check_dtype:
        assert a1_numpy.dtype == a2_numpy.dtype, (
            a1_numpy.dtype, a2_numpy.dtype)
        assert a1_numpy.strides == a2_numpy.strides, (
            a1_numpy.strides, a2_numpy.strides)
    assert a1_numpy.shape == a2_numpy.shape, (a1_numpy.shape, a2_numpy.shape)

    if not np.allclose(a1_numpy, a2_numpy, rtol=rtol, atol=atol):
        raise Exception(
            f"mean_diff: {np.abs(a1_numpy.astype(float) - a2_numpy.astype(float)).mean()}\n"
            f"expected : {a1_numpy.flatten()[:10]}\n"
            f"found    : {a2_numpy.flatten()[:10]}\n"
        )


def from_torch(a: torch.Tensor):
    return Tensor.from_numpy(a.detach().numpy()).requires_grad_(a.requires_grad)


def from_numpy(a: np.typing.NDArray): return Tensor.from_numpy(a)
