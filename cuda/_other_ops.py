
import ctypes
from ctypes import c_int, c_void_p
from .utils import _define_func, _int_1d_array, promote_dtype


SETITEM_DTYPES = [
    "float32", "float64", "int32", "int64",
]


def setitem_op_name(dtype):
    return f"setitem_{dtype}"


def define_setitem_op(lib: ctypes.CDLL, name: str):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(),  # values, stride
        c_void_p,  # condition
        c_void_p, _int_1d_array(), _int_1d_array(),  # out,shape,stride
        c_int,
    ])


def setitem_code(dtype: str):
    func_name = setitem_op_name(dtype)
    code = f"""
        extern "C" void {func_name}(
                const {dtype}* Values, const int* stride_Val,
                const bool *Condition,
                {dtype}* Out, const int* shape_Out,const int* stride_Out,
                int ndim
        ){{
            set_item(
                Values, stride_Val,
                Condition,
                Out, shape_Out, stride_Out,
                ndim
            );
        }}
        """
    return code


def other_ops_source_code():
    return "\n\n".join(map(lambda v: setitem_code(v), SETITEM_DTYPES))


def register_other_ops(lib: ctypes.CDLL, ops: dict):
    for dtype in SETITEM_DTYPES:
        opname = setitem_op_name(dtype)
        ops[opname] = define_setitem_op(lib, opname)
