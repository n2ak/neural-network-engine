
import ctypes
from ctypes import c_int, c_void_p
from .utils import _define_func, _int_1d_array

REDUCE_OPS = [
    ("sum",
        ("float32", "float32"), ("float64", "float64"),
        ("int32", "int64"), ("int64", "int64")
     ),
    ("max",
        ("float32", "float32"), ("float64", "float64"),
        ("int32", "int32"), ("int64", "int64")),
]


def recuceop_name(name, in_dtype, out_dtype):
    return f"reduce_{name}_{in_dtype}_{out_dtype}"


def define_reduce_op(lib: ctypes.CDLL, name: str):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(), _int_1d_array(),  # a
        c_void_p, _int_1d_array(), _int_1d_array(),  # output
        _int_1d_array(),
        c_int,
        c_int,
        c_int,
    ], None)


def register_reduce_ops(lib: ctypes.CDLL, ops: dict):
    for name, *dtypes in REDUCE_OPS:
        for (in_dtype, out_dtypes) in dtypes:
            opname = recuceop_name(name, in_dtype, out_dtypes)
            ops[opname] = define_reduce_op(lib, opname)


def recude_code(name, *dtypes: tuple[str, str]):
    op = name

    def recudeop(types: tuple[str, str]):
        input_dtype, out_dtype = types
        func_name = recuceop_name(name, input_dtype, out_dtype)
        kernel_name = f"{func_name}_kernel"
        code = f"""
extern "C" __global__
void {kernel_name}(
    const {input_dtype}* input,
    {out_dtype}* output,
    const int* in_shape,
    const int* in_strides,
    const int* out_shape,
    const int* out_strides,
    const int* axes,
    int nreduce_axes,
    int ndim_in,
    int ndim_out,
    int out_size
){{
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_idx >= out_size) return;

    int out_coords[MAX_DIMS];
    int tmp = out_idx;
    for (int d = 0; d < ndim_out; ++d) {{
        out_coords[d] = tmp / out_strides[d];
        tmp %= out_strides[d];
    }}

    int in_coords[16];
    int j = 0;
    for (int d = 0; d < ndim_in; ++d) {{
        bool is_reduced = false;
        for (int r = 0; r < nreduce_axes; ++r) {{
            if (axes[r] == d) {{ is_reduced = true; break; }}
        }}
        if (is_reduced) {{
            in_coords[d] = 0;
        }} else {{
            in_coords[d] = out_coords[j++];
        }}
    }}

    {out_dtype} acc = 0.0f;

    int total_reduce = 1;
    for (int r = 0; r < nreduce_axes; ++r) {{
        total_reduce *= in_shape[axes[r]];
    }}

    for (int r = 0; r < total_reduce; ++r) {{
        int rem = r;
        for (int k = 0; k < nreduce_axes; ++k) {{
            int ax = axes[k];
            int size = in_shape[ax];
            int coord = rem % size;
            rem /= size;
            in_coords[ax] = coord;
        }}

        int in_off = 0;
        for (int d = 0; d < ndim_in; ++d) {{
            in_off += in_coords[d] * in_strides[d];
        }}
        acc = {op}(acc,({out_dtype}) input[in_off]);
    }}

    output[flattenIndex(ndim_out, out_coords, out_strides)] = acc;
}}

extern "C" void
{func_name}(
    const {input_dtype}* A, const int* stride_A, const int* shape_A,
    {out_dtype}* C, const int* stride_C, const int* shape_C,
    const int* axis,
    int ndim_A, int ndim_C,
    int nAxis,
    int keepdim
){{
    int totalSize = _size(shape_C, ndim_C);

    int *d_axis; shapeToDevice(axis, &d_axis, ndim_A);

    int *d_shape_A; shapeToDevice(shape_A, &d_shape_A, ndim_A);
    int *d_shape_C; shapeToDevice(shape_C, &d_shape_C, ndim_A);

    int *d_stride_A; shapeToDevice(stride_A, &d_stride_A, ndim_A);
    int *d_stride_C; shapeToDevice(stride_C, &d_stride_C, ndim_A);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(
        A,
        C,
        d_shape_A,
        d_stride_A,
        d_shape_C,
        d_stride_C,
        d_axis,
        nAxis,
        ndim_A,
        ndim_C,
        totalSize
    );
}}
"""
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(recudeop, dtypes))
