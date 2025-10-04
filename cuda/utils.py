import os
import tempfile
import subprocess
import numpy as np


def compile_cuda_code(code: str, lib_name="libtemp.so"):
    nvcc_path = os.getenv("NVCC_PATH", "/usr/local/cuda-12.8/bin/nvcc")
    assert os.path.exists(nvcc_path), "NVCC not found"

    with tempfile.TemporaryDirectory(delete=False) as tmpdir:
        cu_path = os.path.join(tmpdir, "kernel.cu")
        so_path = os.path.join(tmpdir, lib_name)
        with open(cu_path, "w") as f:
            f.write(code)
        cmd = [
            nvcc_path,
            "-shared",
            "-Xcompiler",
            "-fPIC",
            cu_path,
            "-o",
            so_path,
            "-lcudart",
            "--expt-relaxed-constexpr",  # for host constexpr to be used is device
            "-arch=sm_86",
        ]
        subprocess.check_call(cmd)
    return so_path


def get_cuda_code():
    from .ops.other_ops import other_ops_source_code
    from .ops.unary_ops import unary_ops_source_code
    from .ops.reduce_ops import reduction_ops_source_code
    from .ops.elemwsie_ops import element_wise_source_code

    code = "\n".join(
        [
            read_cuda_source(),
            reduction_ops_source_code(),
            element_wise_source_code(),
            unary_ops_source_code(),
            other_ops_source_code(),
        ]
    )
    return code


def _int_1d_array():
    return np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")


def assert_cuda_error(err):
    assert err == 0, err


def promote_dtype(dtype1, dtype2, floating_op: bool) -> str:
    dt1 = np.dtype(dtype1)
    dt2 = np.dtype(dtype2)
    if np.issubdtype(dt1, np.floating) or np.issubdtype(dt2, np.floating):
        if dt1 == np.float64 or dt2 == np.float64:
            return "float64"
        return "float32"
    if floating_op:
        return "float32"
    if dt1 == np.int64 or dt2 == np.int64:
        return "int64"
    return dtype1 if dt1.itemsize >= dt2.itemsize else dtype2


def promote_uop_dtype(input_dtype, floating_operation):
    input_dtype = np.dtype(input_dtype)
    if floating_operation:
        out_dtype = "float32"
        if input_dtype == "float64":
            out_dtype = "float64"
    else:
        out_dtype = input_dtype
    return np.dtype(out_dtype)


def read_cuda_source():
    import pathlib

    source_dir = pathlib.Path(__file__).parent / "csrc"
    with open(source_dir / "header.cuh") as f:
        header = f.read()
    source = header + "\n\n"

    for file in source_dir.glob("*.cu"):
        with open(file) as f:
            source += f.read() + "\n\n"
    return source
