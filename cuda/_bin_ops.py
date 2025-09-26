
import ctypes
from ctypes import c_int, c_void_p
from .utils import _define_func, _int_1d_array, promote_dtype

BIN_OPS = [
    ("add", False, "float32", "float64", "int32", "int64"),
    ("sub", False, "float32", "float64", "int32", "int64"),
    ("mul", False, "float32", "float64", "int32", "int64"),
    ("div", True, "float32", "float64", "int32", "int64"),
]


def bin_op_name(name, in_dtype, in_dtype2, out_dtype):
    return f"bin_{name}_{in_dtype}_{in_dtype2}_{out_dtype}"


def define_bin_op(lib: ctypes.CDLL, name: str, type):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(),  # a , stride of a
        type,  # b
        c_void_p, _int_1d_array(),  # output,stride of output
        _int_1d_array(),  # shape
        c_int,
    ], None)


def register_bin_ops(lib: ctypes.CDLL, ops: dict):
    for name, floating_op, *dtypes in BIN_OPS:
        for in_dtype1 in dtypes:
            for in_dtype2 in ["float32", "int32"]:
                ctype = {
                    "float32": ctypes.c_float,
                    "int32": ctypes.c_int32,
                }[in_dtype2]
                out_dtype = promote_dtype(in_dtype1, in_dtype2, floating_op)
                opname = bin_op_name(name, in_dtype1, in_dtype2, out_dtype)
                ops[opname] = define_bin_op(lib, opname, ctype)


def bin_ops_code(name, floating_op, *dtypes: str):
    op = "_" + name

    def bin_op(input_dtype1, input_dtype2):
        out_dtype = promote_dtype(input_dtype1, input_dtype2, floating_op)
        func_name = bin_op_name(name, input_dtype1, input_dtype2, out_dtype)
        kernel_name = f"{func_name}_kernel"
        code = f"""
extern "C" __global__
void {kernel_name}(
    const {input_dtype1}* A, const int* stride_A,
    const {input_dtype2} B,
    {out_dtype}* C, const int* stride_C,
    const int* shape,
    int ndim,
    int totalSize
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape, ndim, coords);

    int flat_A = flattenIndex(ndim, coords, stride_A);
    int flat_C = flattenIndex(ndim, coords, stride_C);

    C[flat_C] = {op}(({out_dtype})A[flat_A],({out_dtype})B);
}}
extern "C" void
{func_name}(
    const {input_dtype1} *d_a, const int* stride_A,
    {input_dtype2} d_b,
    {out_dtype} *d_c, const int* stride_C,
    int* shape,
    int ndim
){{
    int totalSize = _size(shape,ndim);

    int *d_shape; shapeToDevice(shape,&d_shape,ndim);
    int *d_stride_A; shapeToDevice(stride_A,&d_stride_A,ndim);
    int *d_stride_C; shapeToDevice(stride_C,&d_stride_C,ndim);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(
        d_a, d_stride_A,
        d_b,
        d_c, d_stride_C,
        d_shape,
        ndim,
        totalSize
    );
}}
    """
        return code
    assert len(dtypes) > 0
    return "\n\n".join([bin_op(d1, d2) for d1 in dtypes for d2 in ["int32", "float32"]])
