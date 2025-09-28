
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


def uops_code(name: str, floating_operation: bool, *dtypes: str):
    op_num = f"_UOP_{name.upper()}"

    def uop(input_dtype: str):
        out_dtype = promote_uop_dtype(input_dtype, floating_operation)
        func_name = uop_name(name, input_dtype, out_dtype)
        code = f"""
        extern "C" void {func_name}(
            const {input_dtype}* A, const int* stride_A,
            {out_dtype}* C, const int* stride_C,
            int* shape,
            int ndim
        ) {{
            unary_op(A, stride_A, C, stride_C, shape, ndim, {op_num});
        }}
        """
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(uop, dtypes))
