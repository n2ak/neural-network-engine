
template<typename T>
__global__ void copy_out_kernel(
    const T *A, const int * shape_A,const int * stride_A,
    T *C,
    int ndim_A,
    int totalSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;
    
    int coords_A[MAX_DIMS]; unravel_index(idx, shape_A, ndim_A, coords_A);
    int flat_A = flattenIndex(ndim_A, coords_A, stride_A);

    C[idx] = A[flat_A];
}
template<typename T>
void copy_out(
    const T * d_A, const int * shape_A,const int * stride_A,
    T* d_C,
    int ndim_A
){
    int totalSize = _size(shape_A, ndim_A);

    int blockSize = BLOCK_SIZE;
    int gridSize = (totalSize + blockSize - 1) / blockSize;

    int *d_shape_A; shapeToDevice(shape_A,&d_shape_A,ndim_A);
    int *d_stride_A; shapeToDevice(stride_A,&d_stride_A,ndim_A);

    
    copy_out_kernel<<<gridSize,blockSize>>>(
        d_A, d_shape_A, d_stride_A,
        d_C,
        ndim_A,
        totalSize
    );
}
#define COPY_OUT(T)                                                  \
    extern "C" void copy_out_##T(                                     \
        const T * d_A, const int * shape_A,const int * stride_A,    \
        T * d_C,                                                     \
        int ndim_A                                                      \
    ){copy_out(d_A, shape_A, stride_A, d_C, ndim_A);}                    \

COPY_OUT(int32)
COPY_OUT(int64)
COPY_OUT(float32)
COPY_OUT(float64)