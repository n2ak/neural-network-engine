
#define _MAX_REDUCTION 1
#define _SUM_REDUCTION 2

template<typename T>
__device__ inline T __reduce(T x,T y,int reduction){
    switch(reduction){
        case _MAX_REDUCTION: return max(x,y);
        case _SUM_REDUCTION: return x + y;
    }
    return 0; // should never reach
}
template<typename T>
__device__ constexpr inline T __default_value(int reduction){
    switch(reduction){
        case _MAX_REDUCTION: return std::numeric_limits<T>::lowest();
        case _SUM_REDUCTION: return 0;
    }
    return 0; // should never reach
}

template<typename I,typename O>
__global__ void  reduction_axis_kernel(
    const I* input,
    O* output,
    const int* in_shape,
    const int* in_strides,
    const int* out_shape,
    const int* out_strides,
    const int* axes,
    int nreduce_axes,
    int ndim_in,
    int ndim_out,
    int out_size,
    int reduction_op
){
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_idx >= out_size) return;

    int out_coords[MAX_DIMS];
    int tmp = out_idx;
    for (int d = 0; d < ndim_out; ++d) {
        out_coords[d] = tmp / out_strides[d];
        tmp %= out_strides[d];
    }

    int in_coords[MAX_DIMS];
    int j = 0;
    for (int d = 0; d < ndim_in; ++d) {
        bool is_reduced = false;
        for (int r = 0; r < nreduce_axes; ++r) {
            if (axes[r] == d) { is_reduced = true; break; }
        }
        if (is_reduced) {
            in_coords[d] = 0;
        } else {
            in_coords[d] = out_coords[j++];
        }
    }

    O acc = 0.0f;

    int total_reduce = 1;
    for (int r = 0; r < nreduce_axes; ++r) {
        total_reduce *= in_shape[axes[r]];
    }

    for (int r = 0; r < total_reduce; ++r) {
        int rem = r;
        for (int k = 0; k < nreduce_axes; ++k) {
            int ax = axes[k];
            int size = in_shape[ax];
            int coord = rem % size;
            rem /= size;
            in_coords[ax] = coord;
        }

        int in_off = 0;
        for (int d = 0; d < ndim_in; ++d) {
            in_off += in_coords[d] * in_strides[d];
        }
        acc = __reduce(acc, (O)input[in_off], reduction_op);
    }

    output[flattenIndex(ndim_out, out_coords, out_strides)] = acc;
}

template<typename I,typename O>
__global__ void
reduction_kernel(
    const I* A,
    O* C,
    int totalSize,
    int reduction_op
){
    __shared__ O shared[BLOCK_SIZE];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < totalSize) shared[threadIdx.x] = A[idx];
    else                shared[threadIdx.x] = __default_value<O>(reduction_op); // TODO
    
    __syncthreads();
    
    for(int s = blockDim.x /2; s > 0 ; s/=2){ // blockDim.x = BLOCK_SIZE
        if(threadIdx.x < s){
            shared[threadIdx.x] = __reduce(shared[threadIdx.x], shared[threadIdx.x + s], reduction_op);
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0)
        C[blockIdx.x] = shared[0];
}

template<typename I,typename O>
void reduction_axis(
    const I* A, const int* stride_A, const int* shape_A,
    O* C, const int* stride_C, const int* shape_C,
    const int* axis,
    int ndim_A, int ndim_C,
    int nAxis,
    int keepdim,
    int reduction_op
){
    int totalSize = _size(shape_C, ndim_C);
    int *d_axis; shapeToDevice(axis, &d_axis, ndim_A);

    int *d_shape_A; shapeToDevice(shape_A, &d_shape_A, ndim_A);
    int *d_shape_C; shapeToDevice(shape_C, &d_shape_C, ndim_A);

    int *d_stride_A; shapeToDevice(stride_A, &d_stride_A, ndim_A);
    int *d_stride_C; shapeToDevice(stride_C, &d_stride_C, ndim_A);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    reduction_axis_kernel<<<gridSize, blockSize>>>(
        A,
        C,
        d_shape_A,
        d_stride_A,
        d_shape_C,
        d_stride_C,
        d_axis,
        nAxis,
        ndim_A,
        ndim_C,
        totalSize,
        reduction_op
    );
}



template<typename I,typename O>
void reduction(
    const I* A,
    O* C,
    int totalSize,
    int reduction_op
){
    int blockSize = BLOCK_SIZE;
    int gridSize = (totalSize + blockSize - 1) / blockSize;

    O* d_out;
    cudaMalloc(&d_out, gridSize * sizeof(O));


    reduction_kernel<<<gridSize, blockSize>>>(A,d_out, totalSize, reduction_op);
    
    while(gridSize > 1){
        int size = gridSize;
        gridSize = (gridSize + blockSize - 1) / blockSize;
        reduction_kernel<<<gridSize, blockSize>>>(d_out,d_out, size, reduction_op);
    }
    
    cudaMemcpy(C, d_out, 1 * sizeof(O), cudaMemcpyDeviceToHost);
}