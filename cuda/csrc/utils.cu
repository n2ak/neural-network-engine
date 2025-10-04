
template <typename T>
__global__ void
copy_out_indices_kernel(const T *A, const int *shape_A, const int *stride_A,
                        const int **indices, const int *shape_out, T *C,
                        int ndim_A, int totalSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalSize)
    return;

  int coords_out[MAX_DIMS];
  unravel_index(idx, shape_out, ndim_A, coords_out);

  int coords_A[MAX_DIMS];
  for (int d = 0; d < ndim_A; d++) {
    coords_A[d] = indices[d][coords_out[d]];
  }

  int flat_A = flattenIndex(ndim_A, coords_A, stride_A);
  C[idx] = A[flat_A];
}

template <typename T>
void copy_out_indices(const T *d_A, const int *shape_A, const int *stride_A,
                      T *d_C, const int **indices, const int *shape_C,
                      int ndim_A
                      // ,int ndim_C
) {
  int totalSize = _size(shape_A, ndim_A);

  int blockSize = BLOCK_SIZE;
  int gridSize = (totalSize + blockSize - 1) / blockSize;

  int *d_shape_A;
  shapeToDevice(shape_A, &d_shape_A, ndim_A);
  int *d_stride_A;
  shapeToDevice(stride_A, &d_stride_A, ndim_A);

  int *d_shape_C;
  shapeToDevice(shape_C, &d_shape_C, ndim_A);

  const int **d_indices;
  cudaMalloc(&d_indices, ndim_A * sizeof(int *));
  cudaMemcpy(d_indices, indices, ndim_A * sizeof(int *),
             cudaMemcpyHostToDevice);

  copy_out_indices_kernel<<<gridSize, blockSize>>>(
      d_A, d_shape_A, d_stride_A, d_indices, d_shape_C, d_C, ndim_A, totalSize);
}
#define COPY_OUT_INDICES(T)                                                    \
  extern "C" void copy_out_indices_##T(                                        \
      const T *d_A, const int *shape_A, const int *stride_A, T *d_C,           \
      const int **indices, const int *shape_C, int ndim_A) {                   \
    copy_out_indices(d_A, shape_A, stride_A, d_C, indices, shape_C, ndim_A);   \
  }

template <typename I, typename O>
__global__ void copy_out_kernel(const I *A, const int *shape_A,
                                const int *stride_A, O *C, int ndim_A,
                                int totalSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalSize)
    return;

  int coords_A[MAX_DIMS];
  unravel_index(idx, shape_A, ndim_A, coords_A);
  int flat_A = flattenIndex(ndim_A, coords_A, stride_A);

  C[idx] = (O)A[flat_A];
}

template <typename I, typename O>
void copy_out(const I *d_A, const int *shape_A, const int *stride_A, O *d_C,
              int ndim_A) {
  int totalSize = _size(shape_A, ndim_A);

  int blockSize = BLOCK_SIZE;
  int gridSize = (totalSize + blockSize - 1) / blockSize;

  int *d_shape_A;
  shapeToDevice(shape_A, &d_shape_A, ndim_A);
  int *d_stride_A;
  shapeToDevice(stride_A, &d_stride_A, ndim_A);

  copy_out_kernel<<<gridSize, blockSize>>>(d_A, d_shape_A, d_stride_A, d_C,
                                           ndim_A, totalSize);
}
#define COPY_OUT(I, O)                                                         \
  extern "C" void copy_out_##I##_##O(const I *d_A, const int *shape_A,         \
                                     const int *stride_A, O *d_C,              \
                                     int ndim_A) {                             \
    copy_out(d_A, shape_A, stride_A, d_C, ndim_A);                             \
  }

COPY_OUT_INDICES(int32)
COPY_OUT_INDICES(int64)
COPY_OUT_INDICES(float32)
COPY_OUT_INDICES(float64)
