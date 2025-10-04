import ctypes


class Binary:
    def __init__(self, path: str) -> None:
        self.bin = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        self.functions: dict[str, CFunction] = {}

    def define_function(self, name: str, args: list, return_type=None):
        kernel = CFunction(self.bin, name, args, return_type)
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
        self._kernel = _define_func(name, args, return_type)

    def launch(self, *args):
        assert (
            len(args) == self.expected_n_args
        ), f"CFunction {self.name} expects {self.expected_n_args} arguments but {len(args)} were given."
        return self._kernel(*args)

    call = launch
