enum __UOP { _UOP_EXP = 0, _UOP_LOG = 1, _UOP_LOG2 = 2 };

template <typename T> __device__ inline T __exec_op(T x, __UOP uop) {
  switch (uop) {
  case _UOP_EXP:
    return exp(x);
  case _UOP_LOG:
    return log(x);
  case _UOP_LOG2:
    return log2(x);
  }
  return 0; // should never reach
}

template <typename I, typename O>
__global__ void unary_op_kernel(const I *A, const int *stride_A, O *C,
                                const int *stride_C, const int *shape, int ndim,
                                int totalSize, __UOP uop) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalSize)
    return;

  int coords[MAX_DIMS];
  unravel_index(idx, shape, ndim, coords);

  int flat_A = flattenIndex(ndim, coords, stride_A);
  int flat_C = flattenIndex(ndim, coords, stride_C);

  C[flat_C] = __exec_op((O)A[flat_A], uop);
}

template <typename I, typename O>
void unary_op(const I *d_a, const int *stride_A, O *d_c, const int *stride_C,
              int *shape, int ndim, __UOP uop) {
  int totalSize = _size(shape, ndim);
  int *d_shape;
  shapeToDevice(shape, &d_shape, ndim);
  int *d_stride_A;
  shapeToDevice(stride_A, &d_stride_A, ndim);
  int *d_stride_C;
  shapeToDevice(stride_C, &d_stride_C, ndim);

  int blockSize = 256;
  int gridSize = (totalSize + blockSize - 1) / blockSize;
  unary_op_kernel<<<gridSize, blockSize>>>(d_a, d_stride_A, d_c, d_stride_C,
                                           d_shape, ndim, totalSize, uop);
}
