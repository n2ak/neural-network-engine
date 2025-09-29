import nn
import torch as T
from tensor import Tensor
from _utils import check, from_torch


class Model1(T.nn.Module):
    def __init__(self, inc, outc) -> None:
        super().__init__()
        self.seq = T.nn.Sequential(
            T.nn.Linear(inc, 64),
            T.nn.ReLU(),
            T.nn.Linear(64, 64),
            T.nn.ReLU(),
            T.nn.Linear(64, 64),
            T.nn.ReLU(),
            T.nn.Linear(64, outc),
        )

    def forward(self, x):
        return self.seq(x)


class Model2(nn.Module[Tensor]):
    def __init__(self, inc, outc) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(inc, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, outc),
        )

    def forward(self, x):
        return self.seq(x)


def test_nn_forward():
    T.manual_seed(0)
    batch = 10
    inc, outc = 30, 10
    x = T.randn(batch, inc)

    lin1 = Model1(inc, outc)
    lin2 = Model2(inc, outc)
    lin2.load_state(lin1.state_dict())

    res1 = lin1(x)
    res2 = lin2(from_torch(x))

    check(res1, res2)
