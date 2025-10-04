#define TILE_DIM 32

// Kernel for batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
__global__ void matmul_batched_kernel(const float *A, const float *B, float *C,
                                      int BATCH, int M, int K, int N,
                                      int strideA_batch, int strideA_m,
                                      int strideA_k, int strideB_batch,
                                      int strideB_k, int strideB_n,
                                      int strideC_batch, int strideC_m,
                                      int strideC_n) {
  __shared__ float As[TILE_DIM][TILE_DIM];
  __shared__ float Bs[TILE_DIM][TILE_DIM];

  int batch = blockIdx.z;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int col = blockIdx.x * TILE_DIM + threadIdx.x;

  float val = 0.0f;

  for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
    int tiledRow = row;
    int tiledCol = t * TILE_DIM + threadIdx.x;

    // A: (B, M, K)
    if (tiledRow < M && tiledCol < K)
      As[threadIdx.y][threadIdx.x] =
          A[batch * strideA_batch + tiledRow * strideA_m +
            tiledCol * strideA_k];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    int tiledRowB = t * TILE_DIM + threadIdx.y;
    int tiledColB = col;

    // B: (B, K, N)
    if (tiledRowB < K && tiledColB < N)
      Bs[threadIdx.y][threadIdx.x] =
          B[batch * strideB_batch + tiledRowB * strideB_k +
            tiledColB * strideB_n];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
      val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  // TODO: move to the top
  if (row < M && col < N) {
    C[batch * strideC_batch + row * strideC_m + col * strideC_n] = val;
  }
}

extern "C" void matmul_batched(const float *A, const float *B, float *C,
                               const int *stride_A, const int *stride_B,
                               const int *stride_C, int BATCH, int M, int K,
                               int N) {
  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM,
            BATCH);

  int strideA_b = stride_A[0];
  int strideA_m = stride_A[1];
  int strideA_k = stride_A[2];

  int strideB_b = stride_B[0];
  int strideB_k = stride_B[1];
  int strideB_n = stride_B[2];

  int strideC_b = stride_C[0];
  int strideC_m = stride_C[1];
  int strideC_n = stride_C[2];

  matmul_batched_kernel<<<grid, block>>>(
      A, B, C, BATCH, M, K, N, strideA_b, strideA_m, strideA_k, strideB_b,
      strideB_k, strideB_n, strideC_b, strideC_m, strideC_n);
}
