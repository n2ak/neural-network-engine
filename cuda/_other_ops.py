from ctypes import c_int, c_void_p, POINTER
from .utils import _int_1d_array


SETITEM_DTYPES = [
    "float32", "float64", "int32", "int64",
]


def setitem_op_name(dtype):
    return f"setitem_{dtype}"


def define_setitem_op(name: str):
    from . import CUDA_KERNELS
    CUDA_KERNELS.define_function(name, [
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
    return "\n".join([
        setitem_source_code(),
        get_copyout_code(),
    ])


def setitem_source_code():
    return "\n\n".join(map(lambda v: setitem_code(v), SETITEM_DTYPES))


def register_other_ops():
    for dtype in SETITEM_DTYPES:
        opname = setitem_op_name(dtype)
        define_setitem_op(opname)

    define_copyout()
    define_copyout_indices()


def define_copyout():
    from . import CUDA_KERNELS
    for in_dtype in ["float32", "float64", "int32", "int64"]:
        for out_dtype in ["float32", "float64", "int32", "int64"]:
            name = f"copy_out_{in_dtype}_{out_dtype}"
            CUDA_KERNELS.define_function(name, [
                c_void_p,  _int_1d_array(), _int_1d_array(),
                c_void_p,
                c_int,
            ])


def define_copyout_indices():
    from . import CUDA_KERNELS
    for dtype in ["float32", "float64", "int32", "int64"]:
        name = f"copy_out_indices_{dtype}"
        CUDA_KERNELS.define_function(name, [
            # const T * d_A, const int * shape_A, const int * stride_A,
            c_void_p,  _int_1d_array(), _int_1d_array(),
            # T * d_C, const int ** indices, const int * shape_C,
            c_void_p, POINTER(c_void_p), _int_1d_array(),
            # int ndim_A
            c_int,
        ])


def get_copyout_code():
    dtypes = ["float32", "float64", "int32", "int64"]
    return "\n".join([
        f"COPY_OUT({in_dtype},{out_dtype})"
        for in_dtype in dtypes
        for out_dtype in dtypes
    ])
