from __future__ import annotations
from .unary_ops import register_uops, uop_name
from .other_ops import register_other_ops, setitem_op_name
from .elemwsie_ops import register_elemwise_ops, elemwise_op_name
from .reduce_ops import register_reduce_ops, reduceop_name, reduction_op_name
from .bin_ops import define_matmul


def register_ops(lib):
    register_elemwise_ops(lib)
    register_uops(lib)
    register_reduce_ops(lib)
    define_matmul(lib)
    register_other_ops(lib)
