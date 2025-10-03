import numpy as np
import dataclasses
from typing import Generator
from tensor import Tensor


@dataclasses.dataclass
class State:
    m: Tensor
    v: Tensor
    vhatmax: Tensor


class Adam:

    def __init__(
        self,
        params: Generator[Tensor, None, None],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
    ) -> None:
        self._initialized = False
        self._params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas
        self.t = 0

        self.init()

    def step(self):
        self._step()

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def init(self):
        self.params = list(self._params)
        assert len(self.params) > 0

        self.state: list[State] = [
            State(
                Tensor.from_numpy(0.0).astype(np.float32),
                Tensor.from_numpy(0.0).astype(np.float32),
                Tensor.from_numpy(0.0).astype(np.float32),
            )
            for _ in range(len(self.params))
        ]

    def _step(self):

        lr = self.lr
        weight_decay = self.weight_decay
        beta1, beta2 = self.betas
        eps = self.eps
        for state, p in zip(self.state, self.params):
            assert p.requires_grad
            assert p.grad is not None
            g: Tensor = p.grad
            if weight_decay != 0:
                g = g + p * weight_decay
            state.m = state.m * beta1 + g * (1 - beta1)
            state.v = state.v * beta2 + (g**2) * (1 - beta2)
            mhat = state.m / (1 + (beta1 * -1))
            vhat = state.v / (1 + (beta2 * -1))
            change = mhat * lr / (vhat.sqrt() + eps)
            # change = Tensor.from_numpy(
            #     np.clip(change.numpy(), -100, 100))
            p -= change.astype(np.float32)
        self.t += 1
