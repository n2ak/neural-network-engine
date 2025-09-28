
import ctypes
from ctypes import c_int, c_void_p
from .utils import _define_func, _int_1d_array, promote_dtype


ELEMWISE_OPS = [
    ("add", False, "float32", "float64", "int32", "int64"),
    ("sub", False, "float32", "float64", "int32", "int64"),
    ("mul", False, "float32", "float64", "int32", "int64"),
    ("div", True, "float32", "float64", "int32", "int64"),
]


def elemwise_op_name(name, in_dtype, in_dtype2, out_dtype):
    return f"elemwise_{name}_{in_dtype}_{in_dtype2}_{out_dtype}"


def define_elemwise_op(lib: ctypes.CDLL, name: str):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(),  # a
        c_void_p, _int_1d_array(),  # b
        c_void_p, _int_1d_array(),  # output
        _int_1d_array(),  # shape
        c_int,
    ], None)


def register_elemwise_ops(lib: ctypes.CDLL, ops: dict):
    for name, floating_op, *dtypes in ELEMWISE_OPS:
        for in_dtype1 in dtypes:
            for in_dtype2 in dtypes:
                out_dtype = promote_dtype(in_dtype1, in_dtype2, floating_op)
                opname = elemwise_op_name(
                    name, in_dtype1, in_dtype2, out_dtype)
                ops[opname] = define_elemwise_op(lib, opname)


def elemwise_code(name: str, floating_op: bool, *dtypes: str):
    op_num = f"_EW_{name.upper()}"

    def elem_op_code_dype(input_dtype1, input_dtype2):
        out_dtype = promote_dtype(input_dtype1, input_dtype2, floating_op)
        func_name = elemwise_op_name(
            name, input_dtype1, input_dtype2, out_dtype)
        code = f"""
        extern "C" void {func_name}(
            const {input_dtype1} *A, const int* stride_A,
            const {input_dtype2} *B, const int* stride_B,
            {out_dtype} *C, const int* stride_C,
            const int* shape,
            int ndim
        ){{
            element_wise(
                A, stride_A,
                B, stride_B,
                C, stride_C,
                shape,
                ndim,
                {op_num}
            );
        }}
        """
        return code
    assert len(dtypes) > 0

    return "\n\n".join([elem_op_code_dype(d1, d2)for d1 in dtypes for d2 in dtypes])
