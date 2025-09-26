
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

    ops["matmul3D_2d"] = define_matmul(lib)


def define_matmul(lib):
    return _define_func(lib["matmul3D_2d"], [
        c_void_p,  # A
        c_void_p,  # B
        c_void_p,  # C
        _int_1d_array(),  # stride_A
        _int_1d_array(),  # stride_B
        _int_1d_array(),  # stride_C
        c_int, c_int, c_int, c_int,  # int BATCH, int M, int K, int N,
    ], None)


TILE_DIM = 32


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


def matmul_code():
    input_dtype1 = "float32"
    input_dtype2 = "float32"
    out_dtype = "float32"
    funcname = "matmul3D_2d"
    kernel_name = f"{funcname}_kernel"
    code = f"""
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM {TILE_DIM}

// A: (B, M, K) 
// B: (K, N)
// C: (B, M, N) 
__global__ void {kernel_name}(
    const {input_dtype1}*  A,
    const {input_dtype2}*  B,
    {out_dtype}*  C,
    int BATCH, int M, int K, int N,
    int strideA_batch, int strideA_m, int strideA_k,
    int strideB_k, int strideB_n,
    int strideC_batch, int strideC_m, int strideC_n
) {{
    __shared__ {out_dtype} As[TILE_DIM][TILE_DIM];
    __shared__ {out_dtype} Bs[TILE_DIM][TILE_DIM];

    int batch = blockIdx.z;
    int row   = blockIdx.y * TILE_DIM + threadIdx.y;
    int col   = blockIdx.x * TILE_DIM + threadIdx.x;

    {out_dtype} val = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {{
        int tiledRow = row;
        int tiledCol = t * TILE_DIM + threadIdx.x;

        if (tiledRow < M && tiledCol < K)
            As[threadIdx.y][threadIdx.x] =
                A[batch * strideA_batch + tiledRow * strideA_m + tiledCol * strideA_k];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int tiledRowB = t * TILE_DIM + threadIdx.y;
        int tiledColB = col;

        if (tiledRowB < K && tiledColB < N)
            Bs[threadIdx.y][threadIdx.x] =
                B[tiledRowB * strideB_k + tiledColB * strideB_n];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {{
            val += ({out_dtype})As[threadIdx.y][k] * ({out_dtype})Bs[k][threadIdx.x];
        }}

        __syncthreads();
    }}

    if (row < M && col < N) {{
        C[batch * strideC_batch + row * strideC_m + col * strideC_n] = val;
    }}
}}
extern "C" void
{funcname}(
    const {input_dtype1}* A,
    const {input_dtype2}* B,
    {out_dtype}* C,
    const int* stride_A,
    const int* stride_B,
    const int* stride_C,
    int BATCH ,int M, int K, int N
){{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM,
        BATCH
    );
    
    int strideA_b = stride_A[0];
    int strideA_m = stride_A[1];
    int strideA_k = stride_A[2];

    int strideB_k = stride_B[0];    
    int strideB_n = stride_B[1];    

    int strideC_b = stride_C[0];
    int strideC_m = stride_C[1];
    int strideC_n = stride_C[2];
            
    {kernel_name}<<<grid, block>>>(
        A, B, C,
        BATCH, M, K, N,
        strideA_b, strideA_m, strideA_k,
        strideB_k, strideB_n,
        strideC_b, strideC_m, strideC_n
    );
}}
"""
    return code
