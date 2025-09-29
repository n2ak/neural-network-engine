enum _EW_OP {
    _EW_ADD = 0,
    _EW_SUB = 1,
    _EW_MUL = 2,
    _EW_DIV = 3,
    _EW_LT  = 4,
    _EW_LE  = 5,
    _EW_GT  = 6,
    _EW_GE  = 7,
};

template<typename T>
__device__ inline T __exec_ew_op(T x, T y, _EW_OP op){
    switch(op){
        case _EW_ADD:  return x + y;
        case _EW_SUB:  return x - y;
        case _EW_MUL:  return x * y;
        case _EW_DIV:  return x / y;
        case _EW_LT :  return x < y;
        case _EW_LE :  return x <= y;
        case _EW_GT :  return x > y;
        case _EW_GE :  return x >= y;
    }
    return 0; // should never reach
}

template<typename I1,typename I2,typename O>
__global__ void element_wise_kernel(
    const I1* A, const int* stride_A,
    const I2* B, const int* stride_B,
    O* C, const int* stride_C,
    const int* shape,
    int ndim,
    int totalSize,
    _EW_OP op
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape, ndim, coords);

    int flat_A = flattenIndex(ndim, coords, stride_A);
    int flat_B = flattenIndex(ndim, coords, stride_B);
    int flat_C = flattenIndex(ndim, coords, stride_C);


    C[flat_C] = __exec_ew_op((O) A[flat_A] ,(O) B[flat_B], op);
}}

template<typename I1,typename I2,typename O>
void element_wise(
    const I1 *d_a, const int* stride_A,
    const I2 *d_b, const int* stride_B,
    O *d_c, const int* stride_C,
    const int* shape,
    int ndim,
    _EW_OP op
){{
    int totalSize = _size(shape,ndim);
    int *d_shape; shapeToDevice(shape,&d_shape,ndim);
    int *d_stride_A; shapeToDevice(stride_A,&d_stride_A,ndim);
    int *d_stride_B; shapeToDevice(stride_B,&d_stride_B,ndim);
    int *d_stride_C; shapeToDevice(stride_C,&d_stride_C,ndim);

    int blockSize = BLOCK_SIZE;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    element_wise_kernel<<<gridSize, blockSize>>>(
        d_a, d_stride_A,
        d_b, d_stride_B,
        d_c, d_stride_C,
        d_shape,
        ndim,
        totalSize,
        op
    );
}}