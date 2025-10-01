import numpy as np
import nn
import torch as T

from tensor import Tensor
from utils import dataloader
from grad import Grad
from _test_utils import check, from_torch, check_tensor


class TorchModel(T.nn.Module):
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
    batch = 9
    inc, outc = 30, 10
    x = T.randn(batch, inc)
    t = from_torch(x)

    lin1 = TorchModel(inc, outc)
    lin2 = Model2(inc, outc)
    lin2.load_state(lin1.state_dict())

    res1 = lin1(x)
    with Grad.on():
        res2: Tensor = lin2(t)

    check_tensor(
        (lin1.seq._modules["0"].weight,),
        (lin2.seq.layers[0].weight,),
        res1, res2,
        check_grad=True,
        atol=1e-6
    )


def test_minist():
    class Model(nn.Module[Tensor]):
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

    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.int32)  # type: ignore

    model = Model(X.shape[1], 10)
    for batch_X, batch_y in dataloader(X, y):
        batch_X = Tensor.from_numpy(batch_X)
        batch_y = Tensor.from_numpy(batch_y)

        logits = model.forward(batch_X)
        loss = nn.cross_entropy(logits, batch_y)
