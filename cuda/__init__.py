import ctypes
from .utils import compile_cuda_code, get_cuda_code
from ._other_ops import register_other_ops
from ._bin_ops import define_matmul
from ._unary_ops import register_uops
from ._reduce_ops import register_reduce_ops
from ._elemwsie_ops import register_elemwise_ops


class Binary:
    def __init__(self, path: str) -> None:
        self.bin = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        self.functions: dict[str, CFunction] = {}

    def define_function(self, name: str, args: list, return_type=None):
        kernel = CFunction(
            self.bin,
            name,
            args,
            return_type
        )
        self.functions[name] = kernel
        return kernel

    def get(self, name: str):
        return self.functions[name]


class CFunction:
    def __init__(self, lib: ctypes.CDLL, name: str, args: list, return_type) -> None:
        def _define_func(name: str, types, ret=None):
            func = lib[name]
            func.argtypes = types
            func.restype = ret
            return func
        self.expected_n_args = len(args)
        self.name = name
        self.return_type = return_type
        self._kernel = _define_func(
            name,
            args,
            return_type
        )

    def launch(self, *args):
        assert len(
            args) == self.expected_n_args, f"CFunction {self.name} expects {self.expected_n_args} arguments but {len(args)} were given."
        return self._kernel(*args)

    call = launch


CUDA_KERNELS = Binary(compile_cuda_code(get_cuda_code(), "cuda_code"))


def register():
    register_elemwise_ops()
    register_uops()
    register_reduce_ops()
    define_matmul()
    register_other_ops()


register()
