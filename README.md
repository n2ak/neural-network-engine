
A simple DL framework inspired by `PyTorch` with GPU (cuda) acceleration, for education purposes only!


### Arithmetic operations with broadcasting

```python
from tensor import Tensor

a = Tensor.randn(3, 5)
b = Tensor.randn(5,)
(a + b).numpy()
```

### Differentiation

```python
from tensor import Tensor
from grad import Grad
import numpy as np
np.random.seed(0)


a = Tensor.randn(3, 5).requires_grad_(True)
b = Tensor.randn(5,)

with Grad.on():
    # now operations can be differentiable
    c = (a * b)

c.backward(Tensor.ones_like(c))

print(a.grad.numpy())
#   [[ 0.33367434  1.4940791  -0.20515826  0.3130677  -0.85409576]
#   [ 0.33367434  1.4940791  -0.20515826  0.3130677  -0.85409576]
#   [ 0.33367434  1.4940791  -0.20515826  0.3130677  -0.85409576]]

# this will fail because `b` doesn't require gradient
# print(b.grad.numpy())
```

### Neural networks

A feed forward neural network

```python
from tensor import Tensor
from grad import Grad
import nn
import numpy as np
np.random.seed(0)


class FFN(nn.Module[Tensor]):
    def __init__(self, inc, outc) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(inc, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, outc),
        )

    def forward(self, x):
        return self.seq(x)
```

Initialize the modal and the `Adam` optimizer:

```python
from optim import Adam
model = FFN(64, 10)
optimizer = Adam(model.parameters(), lr=.001)
```

Forward pass:

```python
BATCH = 1000
X = Tensor.randn(BATCH, 64)
y = Tensor.randint(0, 10, (BATCH,))

with Grad.on():
    logits = model.forward(X)
    loss = nn.cross_entropy(logits, y)

print(loss.numpy())
#   2.3026597
```

Backward pass:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

for a complele example see: [mnist example](/mnist.ipynb).


### Missing

Some of the missing parts:

* Conv nets and conv operations