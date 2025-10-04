from ctypes import c_int, c_void_p
from ..utils import _int_1d_array

REDUCE_OPS = [
    (
        "sum",
        ("float32", "float32"),
        ("float64", "float64"),
        ("int32", "int64"),
        ("int64", "int64"),
    ),
    (
        "max",
        ("float32", "float32"),
        ("float64", "float64"),
        ("int32", "int32"),
        ("int64", "int64"),
    ),
]


def reduceop_name(name, in_dtype, out_dtype):
    return f"reduce_{name}_{in_dtype}_{out_dtype}"


def reduction_op_name(name, in_dtype, out_dtype):
    return f"reduction_{name}_{in_dtype}_{out_dtype}"


def define_reduce_op(lib, name: str):
    lib.define_function(
        name,
        [
            c_void_p,
            _int_1d_array(),
            _int_1d_array(),  # a
            c_void_p,
            _int_1d_array(),
            _int_1d_array(),  # output
            _int_1d_array(),
            c_int,
            c_int,
            c_int,
        ],
    )


def define_reducction_op(lib, name: str):
    lib.define_function(
        name,
        [
            c_void_p,  # a
            c_void_p,  # output
            c_int,  # totalSize
        ],
    )


def register_reduce_ops(lib):
    for name, *dtypes in REDUCE_OPS:
        for in_dtype, out_dtypes in dtypes:
            define_reduce_op(lib, reduceop_name(name, in_dtype, out_dtypes))
            define_reducction_op(lib, reduction_op_name(name, in_dtype, out_dtypes))


def reduce_axis_code(name: str, *dtypes: tuple[str, str]):
    reduction = f"_{name.upper()}_REDUCTION"

    def reduceop(types: tuple[str, str]):
        input_dtype, out_dtype = types
        func_name = reduceop_name(name, input_dtype, out_dtype)
        code = f"""
        extern "C" void {func_name}(
            const {input_dtype}* A, const int* stride_A, const int* shape_A,
            {out_dtype}* C, const int* stride_C, const int* shape_C,
            const int* axis,
            int ndim_A, int ndim_C,
            int nAxis
        ) {{
            reduction_axis(
                A, stride_A,shape_A,
                C, stride_C, shape_C,
                axis,
                ndim_A, ndim_C,
                nAxis,
                {reduction}
            );
        }}
        """
        return code

    assert len(dtypes) > 0
    return "\n\n".join(map(reduceop, dtypes))


def reduction_op_code(name: str, *dtypes: tuple[str, str]):
    reduction = f"_{name.upper()}_REDUCTION"

    def reduceop(types: tuple[str, str]):
        input_dtype, out_dtype = types
        func_name = reduction_op_name(name, input_dtype, out_dtype)
        # default_value = f"std::numeric_limits<{out_dtype}>::lowest()"
        code2 = f"""
        extern "C" void {func_name}(const {input_dtype}* A,{out_dtype}* C,int totalSize) {{
            reduction(A, C,totalSize,{reduction});
        }}
        """
        return code2

    assert len(dtypes) > 0
    return "\n\n".join(map(reduceop, dtypes))


def reduction_ops_source_code():
    return "\n\n".join(map(lambda v: reduce_axis_code(*v), REDUCE_OPS)) + "\n\n".join(
        map(lambda v: reduction_op_code(*v), REDUCE_OPS)
    )
