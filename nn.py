import numpy as np
from typing import Generic, Type, TypeVar, Any, Generator
from abc import ABC, abstractmethod
from tensor import Tensor
from grad import differentiable

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

    def parameters(self) -> Generator[Tensor, None, None]:
        for attr, value in self.__dict__.items():
            if issubclass(value.__class__, Module):
                yield from value.parameters()
            elif isinstance(value, Tensor) and value.requires_grad:
                yield value

    @property
    def name(self):
        return self.__class__.__name__


class Linear(Module[Tensor]):

    def __init__(self, inc: int, outc: int, bias=True) -> None:
        self.bias = None

        self.weight = kaiming((outc, inc), fan_mode=inc).requires_grad_(True)
        if bias:
            self.bias = kaiming((outc,), fan_mode=inc).requires_grad_(True)

    def forward(self, x):
        x = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias

        return x

    def _load_state(self, member_path: list[str], state):
        assert len(member_path) == 1
        member = member_path[0]

        state = state.numpy()
        if member == "weight":
            state = state  # .T
            old_tensor = self.weight
        elif member == "bias":
            # TODO: check if this module doenst have bias
            old_tensor = self.bias
            assert (
                old_tensor is not None
            ), "This layer doesn't contain a bias, but the state provides it!"
        else:
            raise Exception(f"Invalid attr '{member}'")

        new_tensor = Tensor.from_numpy(state).requires_grad_(True)
        assert (
            old_tensor.shape == new_tensor.shape
        ), f"Expected shape: {old_tensor.shape} but found: {new_tensor.shape}"
        setattr(self, member, new_tensor)


class Sequential(Module[Any]):
    def __init__(self, *layers: Module) -> None:
        self.layers = layers

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer.forward(x)
        return x

    def _load_state(self, member_path: list[str], state):
        assert member_path[0].isdigit()
        index = int(member_path[0])
        self.layers[index]._load_state(member_path[1:], state)

    def parameters(self):
        for l in self.layers:
            yield from l.parameters()


class ReLU(Module[Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        x[x < 0] = 0
        return x


@differentiable(1)
def cross_entropy(input: Tensor, target: Tensor, dim=-1):
    assert input.ndim == 2
    assert target.ndim == 1
    assert input.shape[0] == target.shape[0]

    x = log_softmax(input, dim)
    x = negative_log_likelihood(x, target)
    batch = target.shape[0]

    def backward(gradient: Tensor):
        dx = softmax(input, dim=dim).numpy()
        dx[list(range(batch)), target.numpy().astype(int)] -= 1
        dx /= batch
        return (Tensor.from_numpy(dx * gradient.numpy()),)

    return x, backward


def negative_log_likelihood(tinput: Tensor, ttarget: Tensor):
    # we don't support selecting with lists yet! (input[list1,list2,...])
    input = tinput.numpy()
    target = ttarget.numpy()
    assert np.all(
        input <= 0
    ), f"Not all elements are negative, has NaNs: {np.any(np.isnan(input))}, max: {input.max()}"
    indices = target.astype(int)
    assert (
        input.shape[0] == target.shape[0]
    ), "Input and target should have same batch size."

    # NOTE: input[:, indices] yields wrong result
    res = input[np.arange(target.size), indices] * -1
    return Tensor.from_numpy(np.array(res.mean()))


def log_softmax(x: Tensor, dim=-1) -> Tensor:
    max = x.max(dim, keepdim=True)
    new_x = x - max
    res = new_x - new_x.exp().sum(dim, keepdim=True).log()
    return res


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    m = x - x.max(axis=dim, keepdim=True)
    e = m.exp()
    res = e / e.sum(axis=dim, keepdim=True)
    return res


def kaiming(
    size,
    fan_mode,
    distribution="normal",
    fun="relu",
    dtype=np.float32,
):
    import numpy as np

    gain = get_gain(fun)
    if isinstance(fan_mode, str):
        assert fan_mode in ["fan_in", "fan_out"]
        f_in, f_out = calculate_fans(size)
        fan_mode = f_in if fan_mode == "fan_in" else f_out
    match distribution:
        case "normal":
            std = gain / np.sqrt(fan_mode)
            t = np.random.normal(0, std**2, size)
        case "uniform":
            bounds = gain * np.sqrt(3 / (fan_mode))
            t = np.random.uniform(-bounds, bounds, size)
        case _:
            raise NotImplementedError(f"Unknown distribution: {distribution}")
    return Tensor.from_numpy(t.astype(dtype))


def get_gain(fun: str):
    import numpy as np

    # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
    fun = fun.lower()
    ones = ["conv1d", "conv2d", "conv3d", "sigmoid", "linear", "identity"]
    if fun in ones:
        return 1
    d = {"relu": np.sqrt(2), "selu": 3 / 4, "tanh": 5 / 4}
    return d[fun]


def calculate_fans(shape):
    import numpy as np

    prod = np.prod(shape)
    return prod // shape[1], prod // shape[0]
