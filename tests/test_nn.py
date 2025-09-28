import torch as T
from nn import Linear
from _utils import check, from_torch


def test_nn_forward():
    T.manual_seed(0)

    lin1 = T.nn.Linear(3, 9)
    lin2 = Linear(3, 9)
    lin2.load_state(lin1.state_dict())

    x = T.randn(5, 3)
    res1 = lin1(x)
    res2 = lin2(from_torch(x))

    check(res1, res2)
