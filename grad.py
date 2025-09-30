from __future__ import annotations
import numpy as np
from contextlib import contextmanager
from typing import Any, Callable, TypeVar, TYPE_CHECKING, Generic, ParamSpec, Unpack, TypeVarTuple
T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")
if TYPE_CHECKING:
    from tensor import Tensor
    Func = Callable[P, tuple[Tensor, Callable[[Tensor], tuple[Unpack[Ts]]]]]

    UnaryOpBackwardFn = Callable[[Tensor], tuple[Tensor]]
    ElemWiseBackwardFn = Callable[[Tensor], tuple[Tensor, Tensor]]
    ReduceOpBackwardFn = Callable[[Tensor], tuple[Tensor]]

    UnaryOpBackwardFnWrapper = Callable[[Tensor, Tensor], UnaryOpBackwardFn]
    ElemWiseBackwardFnWrapper = Callable[[
        Tensor, Tensor, Tensor], ElemWiseBackwardFn]

    ReduceOpBackwardFnWrapper = Callable[[
        Tensor, Tensor, tuple[int, ...], bool], ReduceOpBackwardFn]


class Grad:
    _state: list[bool] = []

    @classmethod
    @contextmanager
    def on(cls, val=True):
        cls._state.append(val)
        yield
        # TODO: we might need to try catch
        cls._state.pop()

    @classmethod
    def is_on(cls):
        if len(cls._state):
            return cls._state[-1]
        return False


class DifferentiableFunction():
    def __init__(
        self,
        function: Func,
        name: str,
        n: int,
    ):
        self.function = function
        self.n = n
        self.name = name

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        with Grad.on(False):
            # No gradient inside differentiable functions
            result, backward_fn = self.function(*args, **kwds)
        if Grad.is_on():
            # print("Called", self.name, "Inputs:", args)
            # we setup for backward pass
            # we save only the inputs that require gradient
            self.setup_for_backward_pass(result, backward_fn, args[:self.n])
        return result

    def setup_for_backward_pass(
        self, result: Tensor,
        backward_fn: Callable[[Tensor], tuple[Tensor, ...]],
        args: tuple[Tensor | int | float]
    ):
        from tensor import Tensor
        self._backward_fn = backward_fn
        self.inputs = args
        result.requires_grad = any(map(lambda a: a.requires_grad, args))
        result._set_backward_fn(self)

    def backward(self, incoming_grad: Tensor):
        global indent
        # the chain rule
        from tensor import Tensor
        indent += 1
        # print("\t"*indent, "Chain rule", self.name)
        with Grad.on(False):
            gradients = self._backward_fn(incoming_grad)
            assert len(
                gradients) == self.n, f"Backward function is expected to return {self.n} gradient tensors, but found {len(gradients)}"
            assert all(map(lambda g: isinstance(g, Tensor), gradients)
                       ), f"Backward function is expected to return tensors only"

            for i, input, gradient in zip(range(self.n), self.inputs, gradients):
                if isinstance(input, Tensor) and input.requires_grad:
                    assert gradient.dtype in [
                        np.float32, np.float64], f"Expected {i+1}th gradient to be a    float instead of {gradient.dtype}"
                    # input.grad = gradient
                    # print("\t"*indent, "Passing gradient to", input)
                    input.backward(gradient)
        indent -= 1


indent = 0


def broadcastable(func: Callable[P, Tensor]) -> Callable[P, Tensor]:
    import functools

    @functools.wraps(func)
    def wrapper(x: Tensor, y: Tensor, *args, **kwargs):
        x, y = x.try_broadcast(y)
        return func(x, y, *args, **kwargs)

    return wrapper


def differentiable_function(number_of_args: int):
    import functools

    def register(func: Func):

        @functools.wraps(func)
        def a(*args, **kwargs):
            dfunc = DifferentiableFunction(func, func.__name__, number_of_args)
            return dfunc(*args, **kwargs)
        return a
    return register


def add_backward(x: Tensor, other: Tensor, res: Tensor) -> ElemWiseBackwardFn:
    def backward(gradient: Tensor):
        return gradient, gradient
    return backward


def sub_backward(x: Tensor, other: Tensor, res: Tensor) -> ElemWiseBackwardFn:
    def backward(gradient: Tensor):
        return gradient, gradient*-1
    return backward


def mul_backward(x: Tensor, other: Tensor, res: Tensor) -> ElemWiseBackwardFn:
    def backward(gradient: Tensor):
        return other * gradient, x * gradient
    return backward


def truediv_backward(x: Tensor, other: Tensor, res: Tensor) -> ElemWiseBackwardFn:
    def backward(gradient: Tensor):
        dx = 1 / other * gradient
        dother = gradient * x * -1 / (other*other)
        return dx, dother
    return backward


def exp_backward(x: Tensor, res: Tensor) -> UnaryOpBackwardFn:
    def backward(gradient: Tensor):
        return res * gradient,
    return backward


def log_backward(x: Tensor, res: Tensor) -> UnaryOpBackwardFn:
    def backward(gradient: Tensor):
        return gradient/x,
    return backward


def log2_backward(x: Tensor, res: Tensor) -> UnaryOpBackwardFn:
    def backward(gradient: Tensor):
        return gradient/(x * ln2),
    return backward


ln2 = np.log(2).item()


def sum_backward(x: Tensor, res: Tensor, axis: tuple[int, ...], keepdim: bool) -> ReduceOpBackwardFn:
    axis = tuple(sorted(list(axis)))  # for insert

    def backward(gradient: Tensor):
        gradient_shape = list(gradient.shape)
        if not keepdim:
            for i in axis:
                gradient_shape.insert(i, 1)
            gradient = gradient.view(*gradient_shape)
        for i in axis:
            gradient_shape[i] = x.shape[i]
        return gradient.expand(*gradient_shape),
    return backward
