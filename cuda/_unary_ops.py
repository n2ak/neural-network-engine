
import ctypes
import numpy as np
from ctypes import c_int, c_void_p
from .utils import _define_func, _int_1d_array


UOPS = [
    ("exp", True, "float32", "float64", "int32", "int64"),
    ("log", True, "float32", "float64", "int32", "int64"),
    ("log2", True, "float32", "float64", "int32", "int64"),
]


def uop_name(name, in_dtype, out_dtype):
    return f"uop_{name}_{in_dtype}_{out_dtype}"


def define_uop(lib: ctypes.CDLL, name: str):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(),  # a
        c_void_p, _int_1d_array(),  # output
        _int_1d_array(),  # shape
        c_int,
    ], None)


def register_uops(lib: ctypes.CDLL, ops: dict):
    for name, floating_op, *dtypes in UOPS:
        for in_dtype in dtypes:
            out_dtype = promote_uop_dtype(in_dtype, floating_op)
            opname = uop_name(name, in_dtype, out_dtype)
            ops[opname] = define_uop(lib, opname)


def promote_uop_dtype(input_dtype, floating_operation):
    input_dtype = np.dtype(input_dtype)
    if floating_operation:
        out_dtype = "float32"
        if input_dtype == "float64":
            out_dtype = "float64"
    else:
        out_dtype = input_dtype
    return np.dtype(out_dtype)


def uops_code(name, floating_operation: bool, *dtypes: str):
    op = name

    def uop(input_dtype: str):
        out_dtype = promote_uop_dtype(input_dtype, floating_operation)
        func_name = uop_name(name, input_dtype, out_dtype)
        kernel_name = f"{func_name}_kernel"
        code = f"""
extern "C" __global__
void {kernel_name}(
    const {input_dtype}* A, const int* stride_A,
    {out_dtype}* C, const int* stride_C,
    const int* shape,
    int ndim,
    int totalSize
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS]; unravel_index(idx, shape, ndim, coords);

    int flat_A = flattenIndex(ndim, coords, stride_A);
    int flat_C = flattenIndex(ndim, coords, stride_C);

    C[flat_C] = {op}(({out_dtype})A[flat_A]);
}}
extern "C" void
{func_name}(
    const {input_dtype} *d_a, const int* stride_A,
    {out_dtype} *d_c, const int* stride_C,
    int* shape,
    int ndim
){{
    int totalSize = _size(shape,ndim);
    int *d_shape; shapeToDevice(shape, &d_shape, ndim);
    int *d_stride_A; shapeToDevice(stride_A, &d_stride_A, ndim);
    int *d_stride_C; shapeToDevice(stride_C, &d_stride_C, ndim);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(
        d_a, d_stride_A,
        d_c, d_stride_C,
        d_shape,
        ndim,
        totalSize
    );
}}
    """
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(uop, dtypes))
