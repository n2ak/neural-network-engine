import numpy as np
from typing import Generic, Type, TypeVar, Any
from abc import ABC, abstractmethod
from tensor import Tensor

T = TypeVar("T")


def is_instance_or_subclass(obj: Any, cls: type) -> bool:
    return isinstance(obj, cls) or issubclass(obj.__class__, cls)


class Module(Generic[T], ABC):

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, *args, **kwds) -> T:
        raise NotImplementedError()

    def load_state(self, state: dict[str, Any]):
        for k, v in state.items():
            parent, *path = k.split(".")
            if len(path) == 0:
                self._load_state([parent], v)
            else:
                child = getattr(self, parent)
                assert issubclass(child.__class__, Module), type(child)
                child._load_state(path, v)

    def _load_state(self, member_path: list[str], state):
        raise NotImplementedError(
            f"class {self.__class__.__name__} doesn't implement _load_state"
        )


class Linear(Module[Tensor]):

    def __init__(self, inc: int, outc: int, bias=True) -> None:
        self.bias = None

        self.weight = Tensor.randn(inc, outc)
        if bias:
            self.bias = Tensor.randn(outc)

    def forward(self, x):
        res = x @ self.weight
        if self.bias is not None:
            res = res + self.bias
        return res

    def _load_state(self, member_path: list[str], state):
        assert len(member_path) == 1
        member = member_path[0]

        state = state.numpy()
        if member == "weight":
            state = state.T
            old_tensor = self.weight
        elif member == "bias":
            # TODO: check if this module doenst have bias
            old_tensor = self.bias
            assert old_tensor is not None
        else:
            raise Exception(f"Invalid attr '{member}'")

        new_tensor = Tensor.from_numpy(state)
        assert old_tensor.shape == new_tensor.shape, (
            f"Expected shape: {old_tensor.shape} but found: {new_tensor.shape}")
        setattr(self, member, new_tensor)


class Sequential(Module[Any]):
    def __init__(self, *layers: Module) -> None:
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _load_state(self, member_path: list[str], state):
        assert member_path[0].isdigit()
        index = int(member_path[0])
        self.layers[index]._load_state(member_path[1:], state)


class ReLU(Module[Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        x[x < 0] = 0
        return x


def cross_entropy(input: Tensor, target: Tensor, dim=-1):
    assert input.ndim == 2
    assert target.ndim == 1
    assert input.shape[0] == target.shape[0]
    x = log_softmax(input, dim)
    x = negative_log_likelihood(x.numpy(), target.numpy())
    return x


def negative_log_likelihood(input: np.ndarray, target: np.ndarray):
    assert np.all(input <= 0), input <= 0
    indices = target.astype(int)
    assert input.shape[0] == target.shape[0], "Input and target should have same batch size."
    res = input[:, indices] * -1
    return Tensor.from_numpy(np.array(res.mean()))


def log_softmax(x: Tensor, dim=-1) -> Tensor:
    max = x.max(dim, keepdim=True)
    new_x = x - max
    res = new_x - new_x.exp().sum(dim, keepdim=True).log()
    return res
