
import os
import ctypes
import tempfile
import subprocess


from ._other_ops import register_other_ops, other_ops_source_code
from ._bin_ops import define_matmul, define_copyout
from ._unary_ops import unary_ops_source_code, register_uops
from ._reduce_ops import reduction_ops_source_code, register_reduce_ops
from ._elemwsie_ops import element_wise_source_code, register_elemwise_ops

from .utils import read_cuda_source

nvcc_path = os.getenv("NVCC_PATH", "/usr/local/cuda-12.8/bin/nvcc")
assert os.path.exists(nvcc_path), "NVCC not found"


def get_cuda_code():
    code = "\n".join([
        read_cuda_source(),
        reduction_ops_source_code(),
        element_wise_source_code(),
        unary_ops_source_code(),
        other_ops_source_code(),
    ])
    return code


def compile_cuda_code(code: str, lib_name="libtemp.so"):
    with tempfile.TemporaryDirectory() as tmpdir:
        cu_path = os.path.join(tmpdir, "kernel.cu")
        so_path = os.path.join(tmpdir, lib_name)
        with open(cu_path, "w") as f:
            f.write(code)
        cmd = [
            nvcc_path, "-shared", "-Xcompiler", "-fPIC", cu_path,
            "-o", so_path,
            "-lcudart",
            "--expt-relaxed-constexpr",  # for host constexpr to be used is device
            "-arch=sm_86",
        ]
        subprocess.check_call(cmd)
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    return lib


def compile():
    lib = compile_cuda_code(get_cuda_code(), "cuda_code")
    ops = {}
    register_elemwise_ops(lib, ops)
    register_uops(lib, ops)
    register_reduce_ops(lib, ops)
    ops["matmul_batched"] = define_matmul(lib)
    define_copyout(lib, ops)
    register_other_ops(lib, ops)
    return ops


_cuda_ops = compile()
