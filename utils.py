import numpy as np


class DataLoader:
    def __init__(
        self,
        *arrs: np.typing.NDArray,
        batch_size=64,
        shuffle=True,
    ):
        self.arrs = arrs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dlen = arrs[0].shape[0]
        self.idx = 0
        self._prepare()

    def _prepare(self):
        if self.shuffle:
            perm = np.random.permutation(self.dlen)
            self.arrs = tuple(arr[perm] for arr in self.arrs)
        self.idx = 0

    def __len__(self): return self.dlen // self.batch_size

    def __iter__(self):
        self._prepare()
        return self

    def __next__(self):
        if self.idx >= self.dlen:
            raise StopIteration
        batch = tuple(arr[self.idx:self.idx+self.batch_size]
                      for arr in self.arrs)
        self.idx += self.batch_size
        return batch
